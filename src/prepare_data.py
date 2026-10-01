"""Create a balanced class-folder dataset from the Kaggle download."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_CLASSES = tuple(range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--target-per-class", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_inputs(data_root: Path) -> tuple[Path, Path]:
    labels_path = data_root / "labels.csv"
    images_dir = data_root / "Images"
    if not labels_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {labels_path} and {images_dir}. See data/README.md."
        )
    return labels_path, images_dir


def read_labels(labels_path: Path) -> dict[str, int]:
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or not {"image", "level"}.issubset(rows.fieldnames):
            raise ValueError("labels.csv must contain image and level columns")
        labels = {row["image"]: int(row["level"]) for row in rows}
    if not labels:
        raise ValueError("labels.csv contains no rows")
    return labels


def collect_by_class(images_dir: Path, labels: dict[str, int]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        if path.stem not in labels:
            continue
        grouped[labels[path.stem]].append(path)
    return grouped


def create_balanced_dataset(
    grouped: dict[int, list[Path]], output_dir: Path, target: int, seed: int
) -> dict[int, int]:
    if target < 1:
        raise ValueError("target-per-class must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. Remove it or choose another output directory."
        )

    for class_id in EXPECTED_CLASSES:
        available = len(grouped.get(class_id, []))
        if available < target:
            raise ValueError(
                f"class_{class_id} has {available} images; {target} are required"
            )

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[int] = Counter()
    for class_id in EXPECTED_CLASSES:
        destination = output_dir / f"class_{class_id}"
        destination.mkdir()
        selected = rng.sample(grouped[class_id], target)
        for source in selected:
            shutil.copy2(source, destination / source.name)
            counts[class_id] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    labels_path, images_dir = resolve_inputs(args.data_root)
    labels = read_labels(labels_path)
    grouped = collect_by_class(images_dir, labels)
    available = {class_id: len(grouped.get(class_id, [])) for class_id in EXPECTED_CLASSES}
    selected = create_balanced_dataset(
        grouped, args.output_dir, args.target_per_class, args.seed
    )
    print(json.dumps({"available": available, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
