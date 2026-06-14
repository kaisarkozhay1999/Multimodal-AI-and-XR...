from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PART_B = ROOT / "part_B"
SESSION = "22_assy_0_1_paper_000000_000199"
IMAGE_DIR = PART_B / "paper_tracking" / "sequences" / SESSION / "rgb"
ANNOTATION_DIR = (
    PART_B / "paper_tracking" / "annotations" / SESSION / "accepted"
)
OUTPUT_DIR = PART_B / "data" / "paper_yolo_dataset"
IMAGE_WIDTH = 1280.0
IMAGE_HEIGHT = 720.0
BLOCK_SIZE = 20


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def split_for_frame(frame_index: int) -> str:
    return "val" if (frame_index // BLOCK_SIZE) % 5 == 4 else "train"


def main() -> None:
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    annotations = sorted(ANNOTATION_DIR.glob("*.json"))
    if len(images) != len(annotations) or not images:
        raise RuntimeError(
            f"Expected matching images and masks, found "
            f"{len(images)} images and {len(annotations)} annotations."
        )

    counts: Counter = Counter()
    positive_counts: Counter = Counter()
    for image_path in images:
        frame_index = int(image_path.stem)
        annotation_path = ANNOTATION_DIR / f"{frame_index:06d}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        split = split_for_frame(frame_index)
        stem = f"{SESSION}_{frame_index:06d}"
        link_or_copy(
            image_path,
            OUTPUT_DIR / "images" / split / f"{stem}.jpg",
        )

        label_lines = []
        for item in annotation.get("objects", []):
            polygon = item.get("polygon") or []
            if len(polygon) < 3:
                continue
            coordinates = []
            for x, y in polygon:
                coordinates.extend(
                    [
                        f"{max(0.0, min(1.0, float(x) / IMAGE_WIDTH)):.6f}",
                        f"{max(0.0, min(1.0, float(y) / IMAGE_HEIGHT)):.6f}",
                    ]
                )
            label_lines.append(f"0 {' '.join(coordinates)}")

        label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_lines), encoding="ascii")
        counts[split] += 1
        positive_counts[split] += int(bool(label_lines))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml = "\n".join(
        [
            f"path: {OUTPUT_DIR.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: instruction_paper",
            "",
        ]
    )
    (OUTPUT_DIR / "data.yaml").write_text(data_yaml, encoding="utf-8")
    manifest = {
        "session": SESSION,
        "split": (
            f"{BLOCK_SIZE}-frame temporal blocks; every fifth block is validation"
        ),
        "images": dict(counts),
        "positive_images": dict(positive_counts),
        "negative_images": {
            split: counts[split] - positive_counts[split]
            for split in ("train", "val")
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print(f"Dataset: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
