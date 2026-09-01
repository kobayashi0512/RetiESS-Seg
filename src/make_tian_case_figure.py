"""Render exact lower-/higher-error held-out OCT examples for the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from run_tian_loo import BoundaryNet, augment_training_batch, boundary_loss, pick_device, prepare
from tian_data import load_complete_tian


def refit(volumes, held_out: int, augmented: bool, device: torch.device) -> BoundaryNet:
    torch.manual_seed(20260831 + 100000 + held_out)
    train_ids = [index for index in range(len(volumes)) if index != held_out]
    frames, labels, _ = prepare(volumes, train_ids, device)
    model = BoundaryNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(120):
        epoch_frames, epoch_labels = (augment_training_batch(frames, labels, 6)
                                      if augmented else (frames, labels))
        optimiser.zero_grad(set_to_none=True)
        output, _ = model(epoch_frames)
        loss = boundary_loss(output, epoch_labels)
        loss.backward()
        optimiser.step()
    return model.eval()


def native_case(subject_number: int, predicted: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path("data/tian_octrima/extracted/FiletoShareforPONE") / f"Subject{subject_number}.mat"
    mat = scipy.io.loadmat(path)
    image = np.asarray(mat["volumedata"], dtype=float)
    index = image.shape[2] // 2
    raw = image[:, :, index]
    manual = ((mat["Observer1"][:, index, :] + mat["Observer2"][:, index, :]) / 2.0).T
    predicted_native = F.interpolate(predicted.unsqueeze(0), size=raw.shape[1], mode="linear", align_corners=True).squeeze(0)
    return raw, manual, (predicted_native.cpu().numpy() * (raw.shape[0] - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-results", default="results/tian_nested_augmentation/outer_folds.csv")
    parser.add_argument("--output", default="paper_assets/tian_heldout_cases")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.outer_results).open()))
    selected = [min(rows, key=lambda row: float(row["outer_mae_px"])), max(rows, key=lambda row: float(row["outer_mae_px"]))]
    volumes = load_complete_tian("data/tian_octrima/extracted/FiletoShareforPONE")
    device = pick_device(args.device)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for axis, row, descriptor in zip(axes, selected, ("Lower-error held-out volume", "Higher-error held-out volume")):
        subject_number = int(row["held_out"].removeprefix("Subject"))
        held_out = subject_number - 1
        model = refit(volumes, held_out, row["selected_candidate"] == "augmented", device)
        with torch.no_grad():
            prediction, _ = model(volumes[held_out].frames.to(device))
        raw, manual, prediction_native = native_case(subject_number, prediction[prediction.shape[0] // 2])
        x = np.arange(raw.shape[1])
        axis.imshow(raw, cmap="gray", aspect="auto", origin="upper")
        for boundary in range(8):
            axis.plot(x, manual[boundary], color="#E67E22", linewidth=1.2, alpha=0.9,
                      label="Consensus manual" if boundary == 0 else None)
            axis.plot(x, prediction_native[boundary], color="#0F4D92", linewidth=0.9, alpha=0.9,
                      linestyle="--", label="Nested-selected model" if boundary == 0 else None)
        axis.set_title(f"{descriptor}\n{row['held_out']} | MAE {float(row['outer_mae_px']):.2f} px", fontsize=10)
        axis.set_xlabel("A-scan position")
        axis.set_ylabel("Axial pixel")
        axis.set_xticks([]); axis.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
