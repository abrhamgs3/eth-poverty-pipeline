"""Assemble a single Markdown report summarizing a pipeline run."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict


def build_markdown_report(context: Dict[str, Any], out_path: str) -> str:
    quality = context["quality_metrics"]
    regression = context["regression_results"]
    perf = context["correction_performance"]
    timings = context.get("timings", {})
    cm = quality["Confusion_Matrix"]

    lines = [
        "# Pipeline Run Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Data source: {context['data_source']}",
        "",
        "## Stage 3 - Poverty-Screen Quality (validation split vs. consumption-based truth)",
        "",
        f"- Validation-set size (n): {quality['n']}",
        f"- Accuracy: {quality['Accuracy']:.4f}",
        f"- Cohen's Kappa: {quality['Cohen_Kappa']:.4f}",
        f"- F1: {quality['F1_Score']:.4f}",
        f"- TPR (sensitivity): {quality['True_Positive_Rate_TPR']:.4f}",
        f"- TNR (specificity): {quality['True_Negative_Rate_TNR']:.4f}",
        f"- Confusion matrix: TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}",
        "",
        "## Stage 4 - Downstream Bias Correction (study split, outcome = mean HAZ)",
        "",
        f"- Benchmark estimate (real poverty indicator, D_true): {context['benchmark_beta']:.4f}",
        "",
        "| Model | beta_1 estimate | Std. Error | 95% CI |",
        "|---|---|---|---|",
    ]
    for name, r in regression.items():
        display_name = name.replace("\n", " ")
        lines.append(
            f"| {display_name} | {r['beta']:.4f} | {r['se']:.4f} | [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] |"
        )

    lines += [
        "",
        "## Stage 5 - Correction Performance",
        "",
        f"- Naive-model attenuation: {perf['attenuation_pct']:.1f}%",
        f"- Naive gap to benchmark: {perf['naive_gap_to_benchmark']:.4f}",
        f"- Corrected gap to benchmark: {perf['corrected_gap_to_benchmark']:.4f}",
        f"- Gap closed by correction: {perf['gap_closed_pct']:.1f}%",
        "",
    ]

    if "chart_path" in context:
        rel = os.path.relpath(context["chart_path"], os.path.dirname(out_path))
        lines += ["## Chart", "", f"![Poverty-to-HAZ effect comparison]({rel})", ""]

    if timings:
        lines += ["## Stage Timings", "", "| Stage | Seconds |", "|---|---|"]
        lines += [f"| {stage} | {duration:.3f} |" for stage, duration in timings.items()]
        lines.append("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return out_path
