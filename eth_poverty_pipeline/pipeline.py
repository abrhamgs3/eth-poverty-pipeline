"""Pipeline orchestrator and CLI.

Wires the stages together in order:

    data_extraction -> annotation -> encoding -> evaluation
        -> econometrics (bias correction) -> performance -> reports

Each stage is independently importable and testable - this module just
sequences them and handles the CLI/logging/timing plumbing.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from typing import Any, Dict

import numpy as np

from .config import PipelineConfig
from .data_extraction import load_poverty_panel
from .annotation import MockLLMClient, annotate_batch
from .annotation.clients import LLMClient
from .encoding import encode_predicted_poverty
from .evaluation import evaluate_poverty_classification
from .econometrics import SimpleOLS, run_bootstrap_inference
from .performance import StageTimer, summarize_correction_performance
from .reports import build_markdown_report, plot_treatment_effect_comparison

logger = logging.getLogger("eth_poverty_pipeline")


def _resolve_client(name: str, accuracy: float, api_key: str | None, rng: np.random.Generator) -> LLMClient:
    if name == "mock":
        return MockLLMClient(accuracy=accuracy, rng=rng)
    if name == "openai":
        from .annotation.clients import OpenAIClient

        return OpenAIClient(api_key=api_key)
    raise ValueError(f"Unknown client '{name}'. Choose from: mock, openai.")


def run_full_pipeline(
    config: PipelineConfig, client_name: str = "mock", api_key: str | None = None
) -> Dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    timer = StageTimer()
    out_dir = config.ensure_output_dir()
    data_source = "real ESS panel" if os.path.exists(config.real_data_path) else "synthetic fallback"

    with timer.time("data_extraction"):
        df = load_poverty_panel(config.real_data_path, config.poverty_quantile, rng=rng)
    logger.info("Loaded %d household-wave rows (%s).", len(df), data_source)

    with timer.time("annotation"):
        client = _resolve_client(client_name, config.mock_accuracy, api_key, rng)
        annotations = annotate_batch(df, client, gold_col="D_true")
        df = df.reset_index(drop=True)
        df["predicted_poverty_status"] = annotations["predicted_poverty_status"]
        df["confidence"] = annotations["confidence"]

    with timer.time("encoding"):
        df = encode_predicted_poverty(df, pred_col="predicted_poverty_status")

    val_size = int(len(df) * config.validation_fraction)
    df_val = df.iloc[:val_size].copy()
    df_study = df.iloc[val_size:].copy()
    logger.info("Validation set: %d rows | Study set: %d rows.", len(df_val), len(df_study))

    with timer.time("evaluation"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quality_metrics = evaluate_poverty_classification(df_val, true_col="D_true", pred_col="D_pred")
            for w in caught:
                logger.warning(str(w.message))

    tpr = quality_metrics["True_Positive_Rate_TPR"]
    tnr = quality_metrics["True_Negative_Rate_TNR"]

    with timer.time("econometrics"):
        benchmark_ols = SimpleOLS(df_study["mean_haz"].to_numpy(), df_study["D_true"].to_numpy()).fit()
        naive_ols = SimpleOLS(df_study["mean_haz"].to_numpy(), df_study["D_pred"].to_numpy()).fit()

        corrected = run_bootstrap_inference(
            df_study, y_col="mean_haz", x_noisy_col="D_pred", tpr=tpr, tnr=tnr,
            n_boot=config.n_bootstrap, rng=rng,
        )

    regression_results = {
        "Benchmark\n(Real Poverty Data)": {
            "beta": float(benchmark_ols.beta[1]), "se": float(benchmark_ols.se[1]),
            "ci_lower": float(benchmark_ols.beta[1] - 1.96 * benchmark_ols.se[1]),
            "ci_upper": float(benchmark_ols.beta[1] + 1.96 * benchmark_ols.se[1]),
        },
        "Naive\n(AI Screen)": {
            "beta": float(naive_ols.beta[1]), "se": float(naive_ols.se[1]),
            "ci_lower": float(naive_ols.beta[1] - 1.96 * naive_ols.se[1]),
            "ci_upper": float(naive_ols.beta[1] + 1.96 * naive_ols.se[1]),
        },
        "Corrected\n(Matrix Method)": {
            "beta": corrected["beta_1"]["estimate"], "se": corrected["beta_1"]["se"],
            "ci_lower": corrected["beta_1"]["ci"][0], "ci_upper": corrected["beta_1"]["ci"][1],
        },
    }
    benchmark_beta = regression_results["Benchmark\n(Real Poverty Data)"]["beta"]

    with timer.time("performance"):
        correction_performance = summarize_correction_performance(
            benchmark_beta=benchmark_beta,
            naive_beta=regression_results["Naive\n(AI Screen)"]["beta"],
            corrected_beta=regression_results["Corrected\n(Matrix Method)"]["beta"],
        )

    with timer.time("reports"):
        chart_path = plot_treatment_effect_comparison(
            regression_results, benchmark_beta, os.path.join(out_dir, "poverty_haz_comparison.png"),
        )
        report_context = {
            "quality_metrics": quality_metrics,
            "regression_results": regression_results,
            "correction_performance": correction_performance,
            "benchmark_beta": benchmark_beta,
            "data_source": data_source,
            "timings": timer.as_dict(),
            "chart_path": chart_path,
        }
        report_path = build_markdown_report(report_context, os.path.join(out_dir, "report.md"))

    return {
        "df_study": df_study,
        "quality_metrics": quality_metrics,
        "regression_results": regression_results,
        "correction_performance": correction_performance,
        "data_source": data_source,
        "timings": timer.as_dict(),
        "report_path": report_path,
        "chart_path": chart_path,
    }


def _print_summary(results: Dict[str, Any]) -> None:
    q = results["quality_metrics"]
    print(f"\nData source: {results['data_source']}")
    print("\n--- Poverty-Screen Quality ---")
    print(f"  Accuracy    : {q['Accuracy']:.4f}")
    print(f"  Cohen Kappa : {q['Cohen_Kappa']:.4f}")
    print(f"  TPR / TNR   : {q['True_Positive_Rate_TPR']:.4f} / {q['True_Negative_Rate_TNR']:.4f}")

    print("\n--- Bias Correction (outcome = mean HAZ) ---")
    for name, r in results["regression_results"].items():
        print(f"  {name.replace(chr(10), ' '):26s}: beta = {r['beta']:.4f}  (se = {r['se']:.4f})")

    perf = results["correction_performance"]
    print(f"\n  Naive attenuation   : {perf['attenuation_pct']:.1f}%")
    print(f"  Gap closed          : {perf['gap_closed_pct']:.1f}%")

    print(f"\nReport written to: {results['report_path']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eth-poverty-pipeline",
        description="Poverty-screening annotation -> econometric bias correction pipeline (Ethiopia ESS panel).",
    )
    parser.add_argument("--client", choices=["mock", "openai"], default="mock")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY).")
    parser.add_argument("--data-path", type=str, default=None, help="Override the real-panel parquet path.")
    parser.add_argument("--poverty-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    config = PipelineConfig(seed=args.seed, poverty_quantile=args.poverty_quantile, n_bootstrap=args.n_bootstrap)
    if args.data_path:
        config.real_data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir

    results = run_full_pipeline(config, client_name=args.client, api_key=args.api_key)
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
