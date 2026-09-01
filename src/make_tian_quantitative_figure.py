"""Exact publication-style plots for per-boundary error and ESS perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    boundary = json.loads(Path("results/tian_per_boundary/summary.json").read_text(encoding="utf-8"))["means"]
    perturbation = json.loads(Path("results/tian_loo_60e/statistical_summary.json").read_text(encoding="utf-8"))["perturbations"]
    boundary_values = [boundary[f"boundary_{index}_mae_px"] for index in range(1, 9)]
    order = ["original_10", "append_last_11", "duplicate_all_20", "drop_alternate_5", "drop_central_5"]
    labels = ["Original", "+1 copy", "All copied", "Alternate\ndropped", "Central\ndropped"]
    raw = [perturbation[item]["mean_raw_slice_count"] for item in order]
    ess = [perturbation[item]["mean_ess"] for item in order]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 1.0})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), constrained_layout=True)
    axis = axes[0]
    axis.bar(np.arange(1, 9), boundary_values, color="#0F4D92", width=0.68)
    axis.set_xlabel("Retinal boundary")
    axis.set_ylabel("MAE (native pixels)")
    axis.set_xticks(np.arange(1, 9))
    axis.set_title("A  Per-boundary outer-test error", loc="left", fontweight="bold")
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)

    axis = axes[1]
    x = np.arange(len(order)); width = 0.35
    raw_bars = axis.bar(x - width / 2, raw, width, color="#B8B8B8", label="Raw B-scan count")
    axis.set_ylabel("Raw B-scan count")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 22)
    axis.set_title("B  Raw count versus correlation-aware ESS", loc="left", fontweight="bold")
    twin = axis.twinx()
    ess_bars = twin.bar(x + width / 2, ess, width, color="#3775BA", label="Effective sample size")
    twin.set_ylabel("Effective sample size")
    twin.set_ylim(0, 3.0)
    axis.spines["top"].set_visible(False); twin.spines["top"].set_visible(False)
    axis.legend([raw_bars, ess_bars], ["Raw B-scan count", "Effective sample size"], frameon=False, fontsize=8, loc="upper left")

    output = Path("paper_assets/tian_quantitative_results")
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
