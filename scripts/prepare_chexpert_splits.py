#!/usr/bin/env python3
"""Create reproducible, patient-disjoint train/dev/calibration/test CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re

CLASS_NAMES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_id(path):
    matches = re.findall(r"(?:^|/)(patient\d+)(?:/|$)", path)
    if len(matches) != 1:
        raise ValueError("Path must contain exactly one patient ID")
    return matches[0]


def image_path(root, path):
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Image paths must be relative to the dataset root")
    return Path(root) / relative


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Path", "Frontal/Lateral", *CLASS_NAMES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing CheXpert columns: " + str(path))
        rows = list(reader)
        fields = reader.fieldnames
    if not rows:
        raise ValueError("Empty CSV: " + str(path))
    paths = [row["Path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths in " + str(path))
    for row in rows:
        patient_id(row["Path"])
        if row["Frontal/Lateral"] not in ("Frontal", "Lateral"):
            raise ValueError("Unexpected view label")
        for name in CLASS_NAMES:
            value = row[name].strip()
            if value and float(value) not in (-1.0, 0.0, 1.0):
                raise ValueError("Unexpected CheXpert label value")
    return fields, rows


def prepare_splits(data_root, output_dir, seed=42, dev_fraction=0.05, calibration_fraction=0.05,
                   check_images=False):
    root, output = Path(data_root), Path(output_dir)
    if not (0 < dev_fraction < 1 and 0 < calibration_fraction < 1
            and dev_fraction + calibration_fraction < 1):
        raise ValueError("dev/calibration fractions must be positive and sum to less than 1")
    fields, source = read_rows(root / "train.csv")
    test_fields, test = read_rows(root / "valid.csv")
    if fields != test_fields:
        raise ValueError("train.csv and valid.csv column order differs")
    all_train_patients = {patient_id(row["Path"]) for row in source}
    test_patients = {patient_id(row["Path"]) for row in test}
    if all_train_patients & test_patients:
        raise ValueError("Patient overlap between official train and valid")
    # Only frontal images are used, including when deciding patient counts.
    source = [row for row in source if row["Frontal/Lateral"] == "Frontal"]
    test = [row for row in test if row["Frontal/Lateral"] == "Frontal"]
    if not test or any(row[name].strip() not in ("0", "1", "0.0", "1.0")
                       for row in test for name in CLASS_NAMES):
        raise ValueError("Official frontal test labels must be complete binary labels")
    patients = sorted({patient_id(row["Path"]) for row in source})
    random.Random(seed).shuffle(patients)
    n_dev = max(1, int(len(patients) * dev_fraction))
    n_cal = max(1, int(len(patients) * calibration_fraction))
    if n_dev + n_cal >= len(patients):
        raise ValueError("Too few patients for three nonempty training-source splits")
    assignment = {p: "dev" for p in patients[:n_dev]}
    assignment.update({p: "calibration" for p in patients[n_dev:n_dev + n_cal]})
    assignment.update({p: "train" for p in patients[n_dev + n_cal:]})
    splits = {name: [] for name in ("train", "dev", "calibration")}
    for row in source:
        splits[assignment[patient_id(row["Path"])]].append(row)
    splits["test"] = test
    manifest = {
        "schema_version": 1, "seed": seed, "dev_fraction": dev_fraction,
        "calibration_fraction": calibration_fraction, "view": "Frontal", "class_order": list(CLASS_NAMES),
        "sources": {name: sha256(root / name) for name in ("train.csv", "valid.csv")},
        "selection_split": "dev", "final_evaluation_split": "test", "splits": {},
        "label_sources": {"train_dev_calibration": "report-derived training labels", "test": "official valid expert labels"},
    }
    for name, rows in splits.items():
        if check_images:
            for row in rows:
                path = image_path(root, row["Path"])
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("Missing or empty image: " + str(path))
        counts = {disease: {label: sum(row[disease].strip() in values for row in rows)
                           for label, values in (("positive", ("1", "1.0")), ("negative", ("0", "0.0")),
                                                 ("uncertain", ("-1", "-1.0")), ("missing", ("",)))}
                  for disease in CLASS_NAMES}
        manifest["splits"][name] = {"images": len(rows), "patients": len({patient_id(r["Path"]) for r in rows}),
                                     "labels": counts}
    # Existing split directories are immutable: avoid silently replacing an experiment's partition.
    if output.exists() and any(output.iterdir()):
        previous = json.loads((output / "manifest.json").read_text())
        expected = {key: value for key, value in previous.items() if key != "files"}
        if expected != manifest or any(sha256(output / name) != digest for name, digest in previous["files"].items()):
            raise ValueError("Existing splits differ; choose a new output directory")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifest["files"] = {name + ".csv": sha256(output / (name + ".csv")) for name in splits}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    try:
        result = prepare_splits(args.data_root.expanduser(), args.output_dir.expanduser(), args.seed,
                                args.dev_fraction, args.calibration_fraction, args.check_images)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
