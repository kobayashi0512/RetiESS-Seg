"""Nested volume-level selection of the augmentation protocol on Tian OCTRIMA.

For every outer held-out volume, the plain and training-augmentation candidates
are compared using only the remaining nine volumes (three deterministic inner
groups).  The selected candidate is then refit on all nine outer-training
volumes and evaluated once on the untouched tenth volume.  This prevents the
outer test volume from being used to choose the optimisation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_tian_loo import BoundaryNet, augment_training_batch, boundary_loss, pick_device, prepare
from tian_data import TianVolume, load_complete_tian, write_audit


def fit_and_score(
    volumes: list[TianVolume], train_ids: list[int], evaluation_ids: list[int],
    use_augmentation: bool, epochs: int, seed: int, device: torch.device,
) -> list[float]:
    torch.manual_seed(seed)
    train_frames, train_labels, _ = prepare(volumes, train_ids, device)
    model = BoundaryNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        frames, labels = (augment_training_batch(train_frames, train_labels, 6)
                          if use_augmentation else (train_frames, train_labels))
        prediction, _ = model(frames)
        loss = boundary_loss(prediction, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for index in evaluation_ids:
            volume = volumes[index]
            prediction, _ = model(volume.frames.to(device))
            error = (prediction - volume.consensus.to(device)).abs().mean().item()
            scores.append(error * (volume.native_height - 1))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tian_octrima/extracted/FiletoShareforPONE")
    parser.add_argument("--output", default="results/tian_nested_augmentation")
    parser.add_argument("--inner-epochs", type=int, default=60)
    parser.add_argument("--outer-epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)
    volumes = load_complete_tian(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_audit(volumes, output / "dataset_audit.json")
    rows = []
    for outer in range(len(volumes)):
        inner_ids = [i for i in range(len(volumes)) if i != outer]
        # Rotation gives three non-overlapping validation groups and uses all 9 volumes.
        groups = [inner_ids[0:3], inner_ids[3:6], inner_ids[6:9]]
        inner_mean: dict[str, float] = {}
        for name, use_aug in (("plain", False), ("augmented", True)):
            values = []
            for group_index, validation_ids in enumerate(groups):
                training_ids = [i for i in inner_ids if i not in validation_ids]
                values.extend(fit_and_score(
                    volumes, training_ids, validation_ids, use_aug,
                    args.inner_epochs, args.seed + 1000 * outer + 10 * group_index + int(use_aug), device,
                ))
            inner_mean[name] = float(np.mean(values))
        selected = min(inner_mean, key=inner_mean.get)
        outer_error = fit_and_score(
            volumes, inner_ids, [outer], selected == "augmented", args.outer_epochs,
            args.seed + 100000 + outer, device,
        )[0]
        row = {
            "held_out": volumes[outer].subject_id,
            "inner_plain_mae_px": inner_mean["plain"],
            "inner_augmented_mae_px": inner_mean["augmented"],
            "selected_candidate": selected,
            "outer_mae_px": outer_error,
        }
        rows.append(row)
        with (output / "outer_folds.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    errors = np.array([row["outer_mae_px"] for row in rows])
    summary = {
        "protocol": "nested volume-level selection; no held-out outer volume participates in candidate selection",
        "candidate_protocols": ["plain", "training-only shift/contrast/noise augmentation"],
        "outer_folds": len(rows),
        "mean_mae_px": float(errors.mean()),
        "sd_mae_px": float(errors.std(ddof=1)),
        "augmentation_selected_folds": int(sum(row["selected_candidate"] == "augmented" for row in rows)),
        "config": vars(args),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
