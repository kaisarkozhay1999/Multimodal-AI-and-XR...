from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from psr_geometry import (
    CLASS_NAMES,
    ROOT,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    sequence_dir,
)


DEFAULT_MODEL = (
    ROOT
    / "models"
    / "yolo"
    / "yolo26s_seg_best.pt"
)
DEFAULT_OUTPUT = ROOT / "data" / "geometric_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed-length geometric features from YOLO masks."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", nargs="+", default=TRAIN_SEQUENCES + TEST_SEQUENCES)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_names() -> list[str]:
    names = []
    fields = [
        "present",
        "confidence",
        "cx",
        "cy",
        "width",
        "height",
        "bbox_area",
        "mask_area",
        "aspect_log",
        "sin_2theta",
        "cos_2theta",
        "solidity",
    ]
    for class_name in CLASS_NAMES:
        names.extend(f"{class_name}.{field}" for field in fields)
    for first in range(len(CLASS_NAMES)):
        for second in range(first + 1, len(CLASS_NAMES)):
            prefix = f"{CLASS_NAMES[first]}__{CLASS_NAMES[second]}"
            names.extend(
                [
                    f"{prefix}.both_present",
                    f"{prefix}.dx",
                    f"{prefix}.dy",
                    f"{prefix}.distance",
                    f"{prefix}.sin_2delta_theta",
                    f"{prefix}.cos_2delta_theta",
                ]
            )
    return names


def mask_geometry(
    polygon: np.ndarray | None,
    bbox: np.ndarray,
    confidence: float,
    width: int,
    height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    bbox_area = box_width * box_height
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    mask_area = bbox_area
    theta = 0.0
    solidity = 1.0
    if polygon is not None and len(polygon) >= 3:
        points = polygon.astype(np.float32)
        area = abs(float(cv2.contourArea(points)))
        if area > 1.0:
            mask_area = area
            moments = cv2.moments(points)
            if moments["m00"]:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            centered = points - points.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(1, len(centered) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            direction = eigenvectors[:, int(np.argmax(eigenvalues))]
            theta = float(np.arctan2(direction[1], direction[0]))
            hull_area = abs(float(cv2.contourArea(cv2.convexHull(points))))
            solidity = area / hull_area if hull_area > 1.0 else 1.0
    return np.asarray(
        [
            1.0,
            confidence,
            cx / width,
            cy / height,
            box_width / width,
            box_height / height,
            bbox_area / (width * height),
            mask_area / (width * height),
            np.log(box_width / box_height),
            np.sin(2.0 * theta),
            np.cos(2.0 * theta),
            np.clip(solidity, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def result_features(result, width: int, height: int) -> np.ndarray:
    per_class = np.zeros((len(CLASS_NAMES), 12), dtype=np.float32)
    polygons = result.masks.xy if result.masks is not None else []
    boxes = result.boxes
    if boxes is not None:
        candidates: list[list[np.ndarray]] = [[] for _ in CLASS_NAMES]
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())
            polygon = polygons[index] if index < len(polygons) else None
            bbox = boxes.xyxy[index].detach().cpu().numpy()
            candidates[class_id].append(
                mask_geometry(polygon, bbox, confidence, width, height)
            )

        # The base anchors the assembly. For every other class, select the
        # detected instance nearest to the base instead of a loose part with a
        # higher YOLO confidence elsewhere on the table.
        if candidates[0]:
            per_class[0] = max(candidates[0], key=lambda item: item[1])
            base_center = per_class[0, 2:4]
        else:
            base_center = None
        for class_id in range(1, len(CLASS_NAMES)):
            if not candidates[class_id]:
                continue
            if base_center is None:
                per_class[class_id] = max(
                    candidates[class_id], key=lambda item: item[1]
                )
            else:
                per_class[class_id] = min(
                    candidates[class_id],
                    key=lambda item: float(
                        np.linalg.norm(item[2:4] - base_center)
                    ),
                )

    relative = []
    for first in range(len(CLASS_NAMES)):
        for second in range(first + 1, len(CLASS_NAMES)):
            both = per_class[first, 0] * per_class[second, 0]
            if both:
                dx = per_class[second, 2] - per_class[first, 2]
                dy = per_class[second, 3] - per_class[first, 3]
                sin_first, cos_first = per_class[first, 9:11]
                sin_second, cos_second = per_class[second, 9:11]
                sin_delta = sin_second * cos_first - cos_second * sin_first
                cos_delta = cos_second * cos_first + sin_second * sin_first
                relative.extend(
                    [both, dx, dy, np.hypot(dx, dy), sin_delta, cos_delta]
                )
            else:
                relative.extend([0.0] * 6)
    return np.concatenate(
        [per_class.reshape(-1), np.asarray(relative, dtype=np.float32)]
    )


def predict_batches(model: YOLO, paths: list[Path], args: argparse.Namespace):
    for start in range(0, len(paths), args.batch):
        batch = paths[start : start + args.batch]
        yield from model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            device=0,
            stream=False,
            verbose=False,
        )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = feature_names()
    model = YOLO(str(args.model.resolve()))
    if [model.names[index] for index in range(len(model.names))] != CLASS_NAMES:
        raise ValueError(f"Unexpected model classes: {model.names}")

    for sequence in args.sequences:
        destination = output / f"{sequence}.npz"
        if destination.exists() and not args.overwrite:
            print(f"{sequence}: already exists, skipping", flush=True)
            continue
        image_paths = sorted((sequence_dir(sequence) / "rgb").glob("*.jpg"))
        first = cv2.imread(str(image_paths[0]))
        height, width = first.shape[:2]
        features = np.empty((len(image_paths), len(names)), dtype=np.float32)
        for index, result in enumerate(predict_batches(model, image_paths, args)):
            features[index] = result_features(result, width, height)
            if index == 0 or (index + 1) % 250 == 0:
                print(f"{sequence}: {index + 1}/{len(image_paths)}", flush=True)
        frame_numbers = np.asarray(
            [int(path.stem) for path in image_paths], dtype=np.int32
        )
        np.savez_compressed(
            destination,
            features=features,
            frame_numbers=frame_numbers,
            frame_names=np.asarray([path.name for path in image_paths]),
        )
        print(f"Saved {destination}", flush=True)

    metadata = {
        "model": str(args.model.resolve()),
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "classes": CLASS_NAMES,
        "feature_count": len(names),
        "feature_names": names,
        "instance_selection": (
            "Highest-confidence base; nearest-to-base instance for every "
            "other class, with confidence fallback when base is absent."
        ),
        "sequences": args.sequences,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
