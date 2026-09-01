"""Loader and integrity audit for the complete public Tian OCTRIMA dataset.

The release contains ten *complete* OCT volumes.  A volume has ten B-scans,
the raw image stack and two independent manual layer-boundary annotations.
This module deliberately does not provide a slice-subset option: every run
loads all ten volumes and patient/volume-disjoint validation is handled by the
caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TianVolume:
    subject_id: str
    frames: torch.Tensor  # [slices, 1, height, width], normalised
    consensus: torch.Tensor  # [slices, boundaries, width], y in [0, 1]
    observer1: torch.Tensor
    observer2: torch.Tensor
    native_height: int
    native_width: int


def _resize_boundary(boundaries: np.ndarray, width: int) -> torch.Tensor:
    """Resize [slices, boundaries, native_width] without changing y values."""
    values = torch.as_tensor(boundaries, dtype=torch.float32)
    return F.interpolate(values, size=width, mode="linear", align_corners=True)


def load_complete_tian(
    root: str | Path,
    image_height: int = 128,
    image_width: int = 256,
) -> list[TianVolume]:
    """Load all ten released volumes; raise if the release is incomplete."""
    base = Path(root)
    paths = sorted(base.glob("Subject*.mat"), key=lambda p: int(p.stem.removeprefix("Subject")))
    expected = [f"Subject{i}.mat" for i in range(1, 11)]
    if [p.name for p in paths] != expected:
        raise RuntimeError(
            "The Tian release must contain exactly Subject1.mat through "
            "Subject10.mat; refusing a partial-dataset experiment."
        )

    volumes: list[TianVolume] = []
    for path in paths:
        mat = scipy.io.loadmat(path)
        required = {"volumedata", "Observer1", "Observer2", "OCTRIMA3D"}
        if missing := required.difference(mat):
            raise RuntimeError(f"{path.name} missing fields: {sorted(missing)}")
        image = np.asarray(mat["volumedata"], dtype=np.float32)  # H, W, S
        obs1 = np.asarray(mat["Observer1"], dtype=np.float32)  # W, S, B
        obs2 = np.asarray(mat["Observer2"], dtype=np.float32)
        if image.ndim != 3 or obs1.ndim != 3 or obs1.shape != obs2.shape:
            raise RuntimeError(f"Unexpected tensor dimensions in {path.name}")
        height, native_width, slices = image.shape
        if obs1.shape != (native_width, slices, 8):
            raise RuntimeError(f"Annotation/image mismatch in {path.name}: {obs1.shape} vs {image.shape}")

        frames = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(1) / 255.0
        frames = F.interpolate(frames, size=(image_height, image_width), mode="bilinear", align_corners=False)
        # Per-slice normalisation prevents scanner exposure from becoming a shortcut.
        mean = frames.mean(dim=(2, 3), keepdim=True)
        std = frames.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        frames = (frames - mean) / std

        def to_normalised(annotation: np.ndarray) -> torch.Tensor:
            # W, S, B -> S, B, W; y values are native-height coordinates.
            boundary = annotation.transpose(1, 2, 0) / float(height - 1)
            return _resize_boundary(boundary, image_width).clamp(0.0, 1.0)

        a1, a2 = to_normalised(obs1), to_normalised(obs2)
        volumes.append(
            TianVolume(
                subject_id=path.stem,
                frames=frames,
                consensus=(a1 + a2) / 2.0,
                observer1=a1,
                observer2=a2,
                native_height=height,
                native_width=native_width,
            )
        )
    return volumes


def write_audit(volumes: list[TianVolume], output: str | Path) -> None:
    """Write a compact provenance and annotation-agreement audit."""
    rows = []
    for volume in volumes:
        agreement_px = (volume.observer1 - volume.observer2).abs().mean().item() * (volume.native_height - 1)
        rows.append(
            {
                "subject_id": volume.subject_id,
                "slices": int(volume.frames.shape[0]),
                "boundaries": int(volume.consensus.shape[1]),
                "native_height": volume.native_height,
                "native_width": volume.native_width,
                "observer1_observer2_mae_px": agreement_px,
            }
        )
    report = {
        "dataset": "Tian OCTRIMA PLOS ONE supplementary release",
        "dataset_policy": "all ten released volumes are loaded; no subject or slice subset is permitted",
        "volume_count": len(rows),
        "slice_count": sum(row["slices"] for row in rows),
        "annotation": "eight retinal layer boundaries, two independent manual observers",
        "volumes": rows,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
