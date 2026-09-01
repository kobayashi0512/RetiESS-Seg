"""Audit the complete anonymous AFIO OCT/fundus archive without modelling it."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook


def clean_name(value: object) -> str:
    name = str(value or "").strip().strip("'").rstrip("'").removesuffix(".jpg")
    # Two spreadsheet cells carry an Excel-style duplicate suffix although the
    # corresponding archive filename does not.  This is a naming repair, not
    # removal of either record.
    return re.sub(r"(?:#1|_1)$", "", name)


def consensus_label(values: list[object]) -> str:
    labels = [str(v).strip().lower() for v in values if v is not None]
    votes = Counter(value for value in labels if value in {"yes", "no"})
    if not votes or len(votes) != 1 and votes["yes"] == votes["no"]:
        return "uncertain"
    winner, count = votes.most_common(1)[0]
    return winner if count >= 2 else "uncertain"


def main() -> None:
    root = Path("data/afio_glaucoma/extracted")
    xlsx = next(root.rglob("*.xlsx"))
    wb = load_workbook(xlsx, data_only=True, read_only=True)
    sheet = wb.active
    rows = []
    for values in sheet.iter_rows(min_row=3, values_only=True):
        image = clean_name(values[0])
        subject = str(values[11] or "").strip()
        if not image or not subject:
            continue
        rows.append({
            "subject": subject.replace(" ", ""),
            "color_stem": image,
            "consensus": consensus_label([values[2], values[4], values[6], values[8]]),
            "grader_votes": [values[2], values[4], values[6], values[8]],
        })
    raw = [path for path in root.rglob("*.jpg") if not path.name.endswith(("_ILM.jpg", "_RPE.jpg"))]
    colors = [path for path in raw if "_Color_" in path.name]
    bscans = [path for path in raw if "_B-scan_" in path.name]
    masks = [path for path in root.rglob("*.jpg") if path.name.endswith(("_ILM.jpg", "_RPE.jpg"))]
    raw_stems = {clean_name(path.stem): path for path in raw}
    unique_rows = {}
    duplicate_metadata = []
    for row in rows:
        if row["color_stem"] in unique_rows:
            duplicate_metadata.append({"subject": row["subject"], "color_stem": row["color_stem"]})
        else:
            unique_rows[row["color_stem"]] = row
    paired, missing_bscan = [], []
    for row in unique_rows.values():
        bscan_stem = row["color_stem"].replace("_Color_", "_B-scan_")
        if bscan_stem in raw_stems and row["color_stem"] in raw_stems:
            paired.append(row)
        else:
            missing_bscan.append({"subject": row["subject"], "color_stem": row["color_stem"], "expected_bscan": bscan_stem})
    patients = defaultdict(list)
    for item in paired:
        patients[item["subject"]].append(item)
    report = {
        "source": "AFIO anonymous public archive, Mendeley dataset 2rnnz5nz74",
        "scope": "complete archive audit; no patient/image subset selected",
        "metadata_rows": len(rows),
        "unique_metadata_image_records": len(unique_rows),
        "duplicate_metadata_records": duplicate_metadata,
        "patient_count": len(patients),
        "raw_color_images": len(colors),
        "raw_bscan_images": len(bscans),
        "layer_mask_images": len(masks),
        "paired_color_oct_records": len(paired),
        "missing_pair_records": missing_bscan,
        "labels": dict(Counter(item["consensus"] for item in paired)),
        "records_per_patient": {key: len(value) for key, value in sorted(patients.items())},
        "patient_disjoint_requirement": "Any later classification split must hold out entire patients, never eyes or images.",
        "clinical_scope_warning": "The archive is a 2D glaucoma-label extension and cannot validate a 3D OCT segmentation claim.",
    }
    destination = Path("results/afio_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
