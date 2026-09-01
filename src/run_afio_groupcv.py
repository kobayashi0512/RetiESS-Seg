"""Complete AFIO patient-disjoint 3-class multimodal extension experiment.

This is explicitly a small 2D paired-image extension.  It is not used as an
external validation of Tian's 3D layer-segmentation result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


CLASSES = ["no", "yes", "uncertain"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


def patient_folds(records: list[dict], n_splits: int = 5) -> list[list[str]]:
    """Deterministic label-aware patient allocation; patients are indivisible."""
    labels = defaultdict(Counter)
    for record in records:
        labels[record["patient_id"]][record["label"]] += 1
    patients = sorted(labels, key=lambda patient: (-sum(labels[patient].values()), patient))
    fold_labels = [Counter() for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    folds = [[] for _ in range(n_splits)]
    for patient in patients:
        vote = labels[patient]
        # Prefer the fold least represented in this patient's observed labels,
        # then the one with fewer images/patients. No images cross folds.
        target = min(range(n_splits), key=lambda fold: (
            sum(fold_labels[fold][label] * count for label, count in vote.items()),
            fold_sizes[fold], len(folds[fold]), fold,
        ))
        folds[target].append(patient)
        fold_labels[target].update(vote)
        fold_sizes[target] += sum(vote.values())
    return folds


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth, pred] += 1
    recalls, f1s = [], []
    for index in range(len(CLASSES)):
        tp = confusion[index, index]
        fn = confusion[index, :].sum() - tp
        fp = confusion[:, index].sum() - tp
        if tp + fn == 0:
            continue  # This class is absent from this held-out patient fold.
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def fit_predict(train_x, train_y, test_x, epochs: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    model = nn.Sequential(nn.Dropout(0.25), nn.Linear(train_x.shape[1], len(CLASSES)))
    counts = torch.bincount(train_y, minlength=len(CLASSES)).float().clamp_min(1)
    weights = counts.sum() / (len(CLASSES) * counts)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y, weight=weights)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        return model(test_x).argmax(dim=1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features/afio_resnet18_complete.pt")
    parser.add_argument("--output", default="results/afio_groupcv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    records = payload["records"]
    if len(records) != 49:
        raise RuntimeError("AFIO group-CV requires all 49 unique paired records")
    color, oct_features = payload["color_features"].float(), payload["oct_features"].float()
    y = torch.tensor([CLASS_TO_INDEX[item["label"]] for item in records], dtype=torch.long)
    folds = patient_folds(records)
    rows = []
    for fold_index, test_patients in enumerate(folds):
        test_set = set(test_patients)
        test_idx = torch.tensor([i for i, item in enumerate(records) if item["patient_id"] in test_set])
        train_idx = torch.tensor([i for i, item in enumerate(records) if item["patient_id"] not in test_set])
        for modality, features in (("fundus", color), ("oct", oct_features), ("fusion", torch.cat([color, oct_features], dim=1))):
            prediction = fit_predict(features[train_idx], y[train_idx], features[test_idx], args.epochs,
                                     args.seed + 100 * fold_index + len(modality))
            row = {
                "fold": fold_index,
                "modality": modality,
                "test_patients": ";".join(sorted(test_patients)),
                "test_records": len(test_idx),
                **metrics(y[test_idx].numpy(), prediction),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fold_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for modality in ("fundus", "oct", "fusion"):
        subset = [row for row in rows if row["modality"] == modality]
        summary[modality] = {
            "folds": len(subset),
            **{metric: float(np.mean([row[metric] for row in subset])) for metric in ("accuracy", "balanced_accuracy", "macro_f1")},
        }
    report = {
        "scope": "complete 49-record, 26-patient AFIO archive; 5-fold patient-disjoint three-class extension",
        "label_order": CLASSES,
        "fold_patient_groups": folds,
        "summary": summary,
        "config": vars(args),
        "warning": "Exploratory 2D multimodal extension; not external validation of 3D OCT segmentation.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
