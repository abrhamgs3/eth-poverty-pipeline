"""Chart generation for the poverty-to-child-health effect comparison.

Imports ``matplotlib`` lazily and forces the ``Agg`` backend before
``pyplot`` is touched, so importing this module never depends on a
display - matters for headless CI and this project's own test suite.

Unlike a pure simulation, there's no known ground-truth effect to draw a
"true value" line at - the reference line here is the benchmark model's
own (real-data) point estimate, itself an estimate with uncertainty, not
an oracle.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def plot_treatment_effect_comparison(
    results: Dict[str, Dict[str, Any]], benchmark_beta: float, out_path: str
) -> str:
    """``results`` maps model name -> {"beta", "se", "ci_lower", "ci_upper"}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    names = list(results.keys())
    betas = [results[m]["beta"] for m in names]
    ci_lowers = [results[m]["ci_lower"] for m in names]
    ci_uppers = [results[m]["ci_upper"] for m in names]
    yerr = [
        [betas[i] - ci_lowers[i] for i in range(len(names))],
        [ci_uppers[i] - betas[i] for i in range(len(names))],
    ]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    ax.axhline(
        benchmark_beta, color="#d62728", linestyle="--", linewidth=2.0, alpha=0.8,
        label=f"Benchmark estimate ({benchmark_beta:.3f})",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)

    data_min = min(ci_lowers + [0.0])
    data_max = max(ci_uppers + [benchmark_beta, 0.0])
    value_range = data_max - data_min
    label_offset = 0.06 * value_range

    for i, name in enumerate(names):
        ax.errorbar(
            x=i, y=betas[i], yerr=[[yerr[0][i]], [yerr[1][i]]], fmt="o", markersize=11,
            capsize=8, color=colors[i], elinewidth=3.0, markeredgecolor="black",
            markeredgewidth=1.5, label=name,
        )
        ax.text(
            i, ci_lowers[i] - label_offset,
            f"{betas[i]:.3f}\n(95% CI: [{ci_lowers[i]:.2f}, {ci_uppers[i]:.2f}])",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2", edgecolor="none"),
        )

    # Explicit ylim, not autoscale: the value-label text sits below the
    # lowest CI bound and autoscale doesn't account for text extent, so
    # without this the label can overflow past the axis into the x-tick
    # labels - which is exactly what happened here with negative effect
    # sizes (the two prior, positive-only demos never exposed this).
    ax.set_ylim(data_min - label_offset - 0.14 * value_range, data_max + 0.08 * value_range)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Estimated Effect of Poverty on Mean HAZ", fontsize=12, fontweight="bold")
    ax.set_title("Correcting for AI-Screening Error in a Poverty-to-Child-Health Estimate", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(-0.5, len(names) - 0.2)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    textstr = "\n".join((
        "Observations:",
        "1. Naive Estimator is attenuated by AI-screening measurement error.",
        "2. Bias Correction rescales the covariance toward the benchmark.",
        "3. The Benchmark itself is a real-data estimate, not a known truth.",
    ))
    props = dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", alpha=0.9, edgecolor="lightgray")
    # Placed below the axes in figure coordinates (fig.text), not inside the
    # data area (ax.text) - the value labels already use the data area's
    # bottom margin, and guessing a data-relative corner that avoids them
    # is exactly what broke last time this used a fixed axes-fraction spot.
    fig.text(0.5, 0.01, textstr, fontsize=9, ha="center", va="bottom", bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
