"""Create a compact, auditable Markdown table from pilot JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("auroc", "auprc", "brier", "nll", "ece_10", "n_eff_mean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot_summary.md"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_dir.glob("*_fold*_fraction*.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "pilot_only_not_publication_result":
            continue
        row = {"file": path.name, "method": result["pooling"],
               "fold": result["test_fold"], "fraction": result["train_fraction"]}
        row.update(result["test"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no pilot results in {args.results_dir}")
    header = ["method", "fold", "fraction", *METRICS, "n_scans", "file"]
    lines = [
        "# CirrusOCT pilot results",
        "",
        "These are preliminary single-seed results, not publication estimates.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        values = []
        for key in header:
            value = row[key]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
