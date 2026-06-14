from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PSR_DATA_ROOT", ROOT)).resolve()
TRAIN_ROOT = DATA_ROOT / "train"
REVIEW_ROOT = DATA_ROOT / "samurai_review" / "data"
DEFAULT_OUTPUT = ROOT / "data" / "yolo10_dataset"

# short_rear_chassis (original class 4) is absent and intentionally excluded.
CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}
NAMES = [
    "base",
    "front_chassis",
    "front_chassis_pin",
    "rear_chassis",
    "front_rear_chassis_pin",
    "rear_rear_chassis_pin",
    "front_bracket",
    "front_bracket_screw",
    "front_wheel_assy",
    "rear_wheel_assy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one completed review sequence as a 10-class YOLO dataset."
    )
    parser.add_argument("--sequence", default="22_assy_0_1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--block-size",
        type=int,
        default=30,
        help="Temporal block size; every tenth block is validation.",
    )
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def split_for_frame(frame_index: int, block_size: int) -> str:
    block_index = frame_index // block_size
    return "val" if block_index % 10 == 9 else "train"


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    image_dir = TRAIN_ROOT / args.sequence / "rgb"
    annotation_dir = REVIEW_ROOT / args.sequence / "accepted"
    image_paths = sorted(image_dir.glob("*.jpg"))
    annotation_paths = sorted(annotation_dir.glob("*.json"))
    if not image_paths or len(image_paths) != len(annotation_paths):
        raise RuntimeError(
            f"Sequence is incomplete: {len(annotation_paths)}/{len(image_paths)}"
        )

    split_counts = Counter()
    class_instances = Counter()
    class_frames = Counter()
    for frame_index, image_path in enumerate(image_paths):
        annotation_path = annotation_dir / f"{frame_index:06d}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        split = split_for_frame(frame_index, args.block_size)
        stem = f"{args.sequence}_{frame_index:06d}"
        link_or_copy(image_path, output / "images" / split / f"{stem}.jpg")

        lines = []
        classes_in_frame = set()
        for obj in annotation.get("objects", []):
            original_id = int(obj["class_id"])
            if original_id not in CLASS_MAP:
                continue
            polygon = obj.get("polygon") or []
            if len(polygon) < 3:
                continue
            class_id = CLASS_MAP[original_id]
            coordinates = []
            for x, y in polygon:
                coordinates.extend(
                    [
                        f"{max(0.0, min(1.0, float(x) / 1280.0)):.6f}",
                        f"{max(0.0, min(1.0, float(y) / 720.0)):.6f}",
                    ]
                )
            lines.append(f"{class_id} {' '.join(coordinates)}")
            class_instances[(split, class_id)] += 1
            classes_in_frame.add(class_id)
        for class_id in classes_in_frame:
            class_frames[(split, class_id)] += 1

        label_path = output / "labels" / split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="ascii")
        split_counts[split] += 1

    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(NAMES))
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    manifest = {
        "sequence": args.sequence,
        "split_method": (
            f"temporal blocks of {args.block_size} frames; every tenth block is val"
        ),
        "class_map": {str(key): value for key, value in CLASS_MAP.items()},
        "names": NAMES,
        "images": dict(split_counts),
        "instances": {
            split: {
                str(class_id): class_instances[(split, class_id)]
                for class_id in range(len(NAMES))
            }
            for split in ("train", "val")
        },
        "frames_with_class": {
            split: {
                str(class_id): class_frames[(split, class_id)]
                for class_id in range(len(NAMES))
            }
            for split in ("train", "val")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Dataset: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
