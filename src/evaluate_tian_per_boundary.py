"""Recover per-boundary outer-test errors from the preregistered nested choices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_tian_loo import BoundaryNet, augment_training_batch, boundary_loss, pick_device, prepare
from tian_data import load_complete_tian


def refit_and_measure(volumes, held_out, augmented, epochs, seed, device):
    train_ids = [index for index in range(len(volumes)) if index != held_out]
    frames, labels, _ = prepare(volumes, train_ids, device)
    torch.manual_seed(seed)
    model = BoundaryNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        epoch_frames, epoch_labels = (augment_training_batch(frames, labels, 6)
                                      if augmented else (frames, labels))
        optimiser.zero_grad(set_to_none=True)
        prediction, _ = model(epoch_frames)
        loss = boundary_loss(prediction, epoch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
    volume = volumes[held_out]
    model.eval()
    with torch.no_grad():
        prediction, _ = model(volume.frames.to(device))
        errors = (prediction - volume.consensus.to(device)).abs().mean(dim=(0, 2))
    return (errors * float(volume.native_height - 1)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-results", default="results/tian_nested_augmentation")
    parser.add_argument("--output", default="results/tian_per_boundary")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)
    volumes = load_complete_tian("data/tian_octrima/extracted/FiletoShareforPONE")
    choices = {row["held_out"]: row["selected_candidate"] for row in csv.DictReader(Path(args.nested_results, "outer_folds.csv").open())}
    rows = []
    for index, volume in enumerate(volumes):
        errors = refit_and_measure(volumes, index, choices[volume.subject_id] == "augmented",
                                   args.epochs, args.seed + 100000 + index, device)
        row = {"held_out": volume.subject_id, "selected_candidate": choices[volume.subject_id],
               **{f"boundary_{i + 1}_mae_px": float(value) for i, value in enumerate(errors)}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_boundary_folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    means = {f"boundary_{i + 1}_mae_px": float(np.mean([row[f"boundary_{i + 1}_mae_px"] for row in rows])) for i in range(8)}
    (output / "summary.json").write_text(json.dumps({"means": means, "folds": len(rows), "source_nested_results": args.nested_results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
