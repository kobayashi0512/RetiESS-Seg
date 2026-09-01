"""Complete-dataset leave-one-volume-out experiment for Tian OCTRIMA.

Two pre-specified training conditions are compared:
* SliceCNN: conventional independent B-scan boundary regression.
* RetiESS-Seg: the same CNN plus a correlation-aware cross-slice relation
  objective.  Its strength is scaled by effective sample size (ESS), so a
  highly redundant volume cannot be treated as ten independent observations.

This is a deliberately compact benchmark.  It is not a disease-classification
claim, and all statistics are at the volume (not B-scan) level.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retiess import RetiESSPool
from tian_data import TianVolume, load_complete_tian, write_audit


class BoundaryNet(nn.Module):
    """Small heatmap regressor; vertical soft-argmax yields eight boundaries."""

    def __init__(self, boundaries: int = 8, feature_dim: int = 64, coordinate_channels: bool = False) -> None:
        super().__init__()
        self.coordinate_channels = coordinate_channels
        input_channels = 2 if coordinate_channels else 1
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, feature_dim, 3, stride=2, padding=1), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(16, boundaries, 4, stride=2, padding=1),
        )

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        if self.coordinate_channels:
            y = torch.linspace(-1, 1, frames.shape[2], dtype=frames.dtype, device=frames.device)
            y = y.view(1, 1, -1, 1).expand(frames.shape[0], 1, -1, frames.shape[3])
            frames = torch.cat([frames, y], dim=1)
        encoded = self.encoder(frames)
        embedding = encoded.mean(dim=(2, 3))
        heatmaps = self.decoder(encoded)
        vertical = torch.softmax(heatmaps, dim=2)
        y = torch.linspace(0, 1, heatmaps.shape[2], device=frames.device, dtype=frames.dtype)
        boundaries = (vertical * y.view(1, 1, -1, 1)).sum(dim=2)
        return boundaries, embedding


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def boundary_loss(prediction: Tensor, target: Tensor, shape_weight: float = 0.0) -> Tensor:
    data = F.smooth_l1_loss(prediction, target)
    # The public annotations are ordered from inner to outer retina.
    order = F.relu(prediction[:, :-1] - prediction[:, 1:] + 0.002).mean()
    shape = F.smooth_l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1],
                             target[:, :, 1:] - target[:, :, :-1])
    return data + 0.25 * order + shape_weight * shape


def relation_loss(prediction: Tensor, target: Tensor, ess_ratio: Tensor) -> Tensor:
    """Cross-slice derivative supervision, downweighted under redundancy."""
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    pred_delta = prediction[1:] - prediction[:-1]
    target_delta = target[1:] - target[:-1]
    return ess_ratio.detach() * F.smooth_l1_loss(pred_delta, target_delta)


def augment_training_batch(
    frames: Tensor, targets: Tensor, max_shift: int,
    use_shift: bool = True, use_appearance: bool = True,
) -> tuple[Tensor, Tensor]:
    """Label-preserving OCT perturbations used on training volumes only."""
    batch, _, height, width = frames.shape
    shifted = []
    shifted_targets = []
    offsets = (torch.randint(-max_shift, max_shift + 1, (batch,), device=frames.device)
               if use_shift and max_shift > 0 else torch.zeros(batch, device=frames.device, dtype=torch.long))
    for index, offset in enumerate(offsets.tolist()):
        item = frames[index:index + 1]
        if offset > 0:
            item = F.pad(item, (0, 0, offset, 0), mode="replicate")[:, :, :height]
        elif offset < 0:
            item = F.pad(item, (0, 0, 0, -offset), mode="replicate")[:, :, -offset:]
        shifted.append(item)
        shifted_targets.append((targets[index:index + 1] + offset / float(height - 1)).clamp(0.0, 1.0))
    augmented = torch.cat(shifted, dim=0)
    adjusted_targets = torch.cat(shifted_targets, dim=0)
    if not use_appearance:
        return augmented, adjusted_targets
    contrast = torch.empty((batch, 1, 1, 1), device=frames.device).uniform_(0.90, 1.10)
    noise = torch.randn_like(augmented) * 0.025
    return augmented * contrast + noise, adjusted_targets


def prepare(volumes: list[TianVolume], ids: list[int], device: torch.device) -> tuple[Tensor, Tensor, list[int]]:
    frames = torch.cat([volumes[i].frames for i in ids]).to(device)
    labels = torch.cat([volumes[i].consensus for i in ids]).to(device)
    counts = [int(volumes[i].frames.shape[0]) for i in ids]
    return frames, labels, counts


def ess_for_volume(pool: RetiESSPool, embeddings: Tensor) -> tuple[Tensor, Tensor]:
    positions = torch.arange(embeddings.shape[0], device=embeddings.device, dtype=embeddings.dtype)
    result = pool(embeddings.unsqueeze(0), positions=positions, scores=torch.zeros(1, embeddings.shape[0], device=embeddings.device))
    return result.n_eff.squeeze(0), result.weights.squeeze(0)


@dataclass
class FoldRow:
    held_out: str
    condition: str
    mae_px: float
    observer1_mae_px: float
    observer2_mae_px: float
    test_ess: float
    duplicate_ess: float


def train_fold(
    volumes: list[TianVolume], held_out: int, condition: str, epochs: int,
    learning_rate: float, seed: int, device: torch.device, model_variant: str,
    shape_weight: float, augment: bool, max_shift: int, use_shift: bool, use_appearance: bool,
) -> FoldRow:
    torch.manual_seed(seed + held_out)
    train_ids = [i for i in range(len(volumes)) if i != held_out]
    train_frames, train_labels, counts = prepare(volumes, train_ids, device)
    model = BoundaryNet(coordinate_channels=(model_variant == "coordconv")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pool = RetiESSPool(feature_dim=64, positional_mix=0.5, position_bandwidth=0.20,
                        feature_bandwidth=0.60, redundancy_strength=1.0,
                        entropy_strength=0.35).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_frames, epoch_labels = (augment_training_batch(train_frames, train_labels, max_shift, use_shift, use_appearance)
                                      if augment else (train_frames, train_labels))
        prediction, embedding = model(epoch_frames)
        loss = boundary_loss(prediction, epoch_labels, shape_weight=shape_weight)
        if condition == "retiess_seg":
            offset = 0
            cross_volume = []
            for count in counts:
                this_pred = prediction[offset:offset + count]
                this_label = epoch_labels[offset:offset + count]
                this_embedding = embedding[offset:offset + count]
                n_eff, _ = ess_for_volume(pool, this_embedding)
                cross_volume.append(relation_loss(this_pred, this_label, n_eff / float(count)))
                offset += count
            loss = loss + 0.35 * torch.stack(cross_volume).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    volume = volumes[held_out]
    model.eval()
    with torch.no_grad():
        frames = volume.frames.to(device)
        predicted, embedding = model(frames)
        consensus = volume.consensus.to(device)
        obs1, obs2 = volume.observer1.to(device), volume.observer2.to(device)
        scale = float(volume.native_height - 1)
        mae = (predicted - consensus).abs().mean().item() * scale
        mae1 = (predicted - obs1).abs().mean().item() * scale
        mae2 = (predicted - obs2).abs().mean().item() * scale
        test_ess, _ = ess_for_volume(pool, embedding)
        duplicate_embedding = torch.cat([embedding, embedding[-1:]], dim=0)
        duplicate_ess, _ = ess_for_volume(pool, duplicate_embedding)
    return FoldRow(volume.subject_id, condition, mae, mae1, mae2, test_ess.item(), duplicate_ess.item())


def write_results(rows: list[FoldRow], output: Path, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "loo_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FoldRow.__annotations__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)
    summary = {}
    for condition in sorted({row.condition for row in rows}):
        group = [row for row in rows if row.condition == condition]
        errors = [row.mae_px for row in group]
        summary[condition] = {
            "folds": len(group),
            "volume_level_mae_px_mean": float(np.mean(errors)),
            "volume_level_mae_px_sd": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
            "mean_test_ess": float(np.mean([row.test_ess for row in group])),
            "mean_duplicate_ess": float(np.mean([row.duplicate_ess for row in group])),
        }
    (output / "summary.json").write_text(json.dumps({"config": vars(args), "summary": summary}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tian_octrima/extracted/FiletoShareforPONE")
    parser.add_argument("--output", default="results/tian_loo")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--model-variant", choices=("plain", "coordconv"), default="plain")
    parser.add_argument("--shape-weight", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument("--disable-shift", action="store_true")
    parser.add_argument("--disable-appearance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    volumes = load_complete_tian(args.data_root, image_height=args.image_height, image_width=args.image_width)
    output = Path(args.output)
    write_audit(volumes, output / "dataset_audit.json")
    rows: list[FoldRow] = []
    for condition in ("slicecnn", "retiess_seg"):
        for held_out in range(len(volumes)):
            row = train_fold(volumes, held_out, condition, args.epochs, args.learning_rate, args.seed, device,
                             args.model_variant, args.shape_weight, args.augment, args.max_shift,
                             not args.disable_shift, not args.disable_appearance)
            rows.append(row)
            print(f"{row.condition} {row.held_out}: MAE={row.mae_px:.3f}px ESS={row.test_ess:.3f}", flush=True)
            write_results(rows, output, args)
    write_results(rows, output, args)


if __name__ == "__main__":
    main()
