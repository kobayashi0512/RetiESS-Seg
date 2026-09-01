"""Cache ungated public ResNet18 features for every AFIO OCT/fundus pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(paths: list[str], model: nn.Module, transform, device: torch.device, batch_size: int) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            values.append(model(torch.stack(batch).to(device)).flatten(1).cpu())
    return torch.cat(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/afio_manifest.json")
    parser.add_argument("--output", default="features/afio_resnet18_complete.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    records = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["records"]
    if len(records) != 49:
        raise RuntimeError("AFIO feature cache requires the complete 49-record manifest")
    device = device_for(args.device)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = nn.Sequential(*list(resnet18(weights=weights).children())[:-1]).to(device).eval()
    color = encode([item["color_path"] for item in records], model, weights.transforms(), device, args.batch_size)
    oct_features = encode([item["oct_path"] for item in records], model, weights.transforms(), device, args.batch_size)
    payload = {
        "source": "complete AFIO 49 unique public pairs; frozen torchvision ResNet18 ImageNet-1K V1",
        "records": records,
        "color_features": color,
        "oct_features": oct_features,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved {len(records)} paired records with {color.shape[1]}-D features to {output}")


if __name__ == "__main__":
    main()
