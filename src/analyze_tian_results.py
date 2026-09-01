"""Volume-level statistics and non-destructive redundancy perturbations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
import torch
import torch.nn.functional as F

from retiess import RetiESSPool
from tian_data import load_complete_tian


def effective_size(features: torch.Tensor, positional_mix: float = 0.5) -> float:
    pool = RetiESSPool(feature_dim=features.shape[1], positional_mix=positional_mix,
                       position_bandwidth=0.20, feature_bandwidth=0.60,
                       redundancy_strength=1.0, entropy_strength=0.35)
    with torch.no_grad():
        result = pool(features.unsqueeze(0), scores=torch.zeros(1, features.shape[0]))
    return float(result.n_eff.item())


def descriptor(frames: torch.Tensor) -> torch.Tensor:
    """Fixed 8x8 image descriptor; no labels, training, or test leakage."""
    return F.adaptive_avg_pool2d(frames, (8, 8)).flatten(1)


def bootstrap_mean_ci(values: np.ndarray, draws: int = 20_000, seed: int = 20260831) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/tian_loo_60e")
    parser.add_argument("--data-root", default="data/tian_octrima/extracted/FiletoShareforPONE")
    args = parser.parse_args()
    root = Path(args.results)
    rows = list(csv.DictReader((root / "loo_results.csv").open(encoding="utf-8")))
    base = {r["held_out"]: float(r["mae_px"]) for r in rows if r["condition"] == "slicecnn"}
    ess = {r["held_out"]: float(r["mae_px"]) for r in rows if r["condition"] == "retiess_seg"}
    ids = sorted(base)
    difference = np.array([ess[x] - base[x] for x in ids])
    test = wilcoxon(difference, alternative="two-sided", method="auto")

    volumes = load_complete_tian(args.data_root)
    perturbation_rows = []
    for volume in volumes:
        feat = descriptor(volume.frames)
        variants = {
            "original_10": feat,
            "append_last_11": torch.cat([feat, feat[-1:]], dim=0),
            "duplicate_all_20": feat.repeat_interleave(2, dim=0),
            "drop_alternate_5": feat[::2],
            "drop_central_5": torch.cat([feat[:2], feat[-3:]], dim=0),
        }
        for name, value in variants.items():
            perturbation_rows.append({
                "subject_id": volume.subject_id,
                "perturbation": name,
                "raw_slice_count": int(value.shape[0]),
                "effective_sample_size": effective_size(value),
            })
    with (root / "redundancy_perturbations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(perturbation_rows[0]))
        writer.writeheader()
        writer.writerows(perturbation_rows)

    by_variant = {}
    for name in sorted({r["perturbation"] for r in perturbation_rows}):
        values = [r["effective_sample_size"] for r in perturbation_rows if r["perturbation"] == name]
        by_variant[name] = {
            "mean_ess": float(np.mean(values)),
            "sd_ess": float(np.std(values, ddof=1)),
            "mean_raw_slice_count": float(np.mean([r["raw_slice_count"] for r in perturbation_rows if r["perturbation"] == name])),
        }
    analysis = {
        "unit_of_inference": "complete OCT volume (n=10), never individual B-scan",
        "primary_comparison": {
            "retiess_minus_slicecnn_mae_px_mean": float(difference.mean()),
            "bootstrap_95pct_ci_px": bootstrap_mean_ci(difference),
            "wilcoxon_two_sided_p": float(test.pvalue),
            "interpretation": "Exploratory fixed-hyperparameter comparison; no claim of a significant segmentation-accuracy gain."
        },
        "perturbation_protocol": "Fixed 8x8 appearance descriptors; uniform slice weights; feature/position kernel mix=0.5. Duplication changes the raw count but should not create independent evidence.",
        "perturbations": by_variant,
    }
    (root / "statistical_summary.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
