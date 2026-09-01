"""Exact three-seed AFIO patient-level exploratory summary plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    sources = [
        Path("results/afio_groupcv_metricfixed/summary.json"),
        Path("results/afio_groupcv_seed_20260901/summary.json"),
        Path("results/afio_groupcv_seed_20260902/summary.json"),
    ]
    summaries = [json.loads(source.read_text(encoding="utf-8"))["summary"] for source in sources]
    modalities = ["fundus", "oct", "fusion"]
    metrics = ["balanced_accuracy", "macro_f1"]
    values = np.array([[[item[modality][metric] for metric in metrics] for modality in modalities] for item in summaries])
    means, stds = values.mean(axis=0), values.std(axis=0, ddof=1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 1.0})
    figure, axis = plt.subplots(figsize=(6.2, 3.5), constrained_layout=True)
    x = np.arange(len(modalities)); width = 0.34
    axis.bar(x - width / 2, means[:, 0], width, yerr=stds[:, 0], capsize=3, color="#3775BA", label="Balanced accuracy")
    axis.bar(x + width / 2, means[:, 1], width, yerr=stds[:, 1], capsize=3, color="#8BCF8B", label="Macro F1")
    axis.set_xticks(x, ["Fundus", "OCT", "Feature fusion"])
    axis.set_ylabel("Patient-disjoint score")
    axis.set_ylim(0, 0.75)
    axis.set_title("AFIO 3-class paired-image extension (3 seeds)", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left")
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    axis.text(0.99, 0.02, "49 records | 26 patients | exploratory", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=8, color="#4D4D4D")
    output = Path("paper_assets/afio_multimodal_summary")
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
