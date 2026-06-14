from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PSR_DATA_ROOT", ROOT / "data" / "labels")).resolve()
TRAIN_ROOT = DATA_ROOT / "train"
DEFAULT_OUTPUT = ROOT / "results" / "yolo_predictions"
SEQUENCES = ["22_assy_2_3", "25_assy_0_1", "25_assy_2_1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 segmentation and render review videos."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback rate. IndustReal RGB frames are sampled at 10 FPS.",
    )
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES)
    return parser.parse_args()


def predict_in_batches(
    model: YOLO,
    image_paths: list[Path],
    imgsz: int,
    batch_size: int,
    conf: float,
):
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        yield from model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=imgsz,
            conf=conf,
            device=0,
            stream=False,
            verbose=False,
        )


def color_for_class(class_id: int) -> tuple[int, int, int]:
    colors = [
        (80, 80, 255),
        (255, 190, 40),
        (40, 210, 255),
        (90, 220, 100),
        (50, 140, 255),
        (210, 210, 60),
        (180, 80, 255),
        (40, 240, 160),
        (40, 180, 255),
        (255, 120, 80),
    ]
    return colors[class_id % len(colors)]


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    x = max(0, min(frame.shape[1] - width - 8, x))
    y = max(height + 7, y)
    cv2.rectangle(
        frame, (x, y - height - 6), (x + width + 7, y + baseline + 2), color, -1
    )
    cv2.putText(
        frame, text, (x + 4, y - 3), font, scale, (10, 10, 10), 1, cv2.LINE_AA
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    model = YOLO(str(args.model.resolve()))
    names = model.names

    for sequence in args.sequences:
        image_paths = sorted((TRAIN_ROOT / sequence / "rgb").glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No images found for {sequence}")
        first = cv2.imread(str(image_paths[0]))
        height, width = first.shape[:2]
        video_dir = output / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{sequence}_yolo26_segmentation.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {video_path}")
        prediction_dir = output / "json" / sequence
        prediction_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = predict_in_batches(
                model=model,
                image_paths=image_paths,
                imgsz=args.imgsz,
                batch_size=args.batch,
                conf=args.conf,
            )
            for frame_index, (image_path, result) in enumerate(zip(image_paths, results)):
                frame = cv2.imread(str(image_path))
                overlay = frame.copy()
                records = []
                polygons = result.masks.xy if result.masks is not None else []
                boxes = result.boxes
                if boxes is not None:
                    for index in range(len(boxes)):
                        class_id = int(boxes.cls[index].item())
                        confidence = float(boxes.conf[index].item())
                        xyxy = boxes.xyxy[index].detach().cpu().numpy().tolist()
                        polygon = (
                            polygons[index].astype(np.int32)
                            if index < len(polygons)
                            else None
                        )
                        color = color_for_class(class_id)
                        if polygon is not None and len(polygon) >= 3:
                            cv2.fillPoly(overlay, [polygon], color)
                        records.append(
                            {
                                "class_id": class_id,
                                "class_name": names[class_id],
                                "confidence": round(confidence, 6),
                                "bbox_xyxy": [round(value, 2) for value in xyxy],
                                "polygon": (
                                    polygon.astype(float).tolist()
                                    if polygon is not None
                                    else []
                                ),
                            }
                        )
                cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
                for record in records:
                    class_id = record["class_id"]
                    color = color_for_class(class_id)
                    polygon = np.asarray(record["polygon"], dtype=np.int32)
                    if len(polygon) >= 3:
                        cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
                    x1, y1, x2, y2 = [
                        int(round(value)) for value in record["bbox_xyxy"]
                    ]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(
                        frame,
                        f"{class_id}: {record['class_name']} {record['confidence']:.2f}",
                        x1,
                        y1 - 3,
                        color,
                    )
                cv2.rectangle(frame, (0, 0), (470, 44), (12, 12, 12), -1)
                cv2.putText(
                    frame,
                    f"{sequence} | frame {frame_index + 1}/{len(image_paths)} | detections {len(records)}",
                    (12, 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                (prediction_dir / f"{frame_index:06d}.json").write_text(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "frame_index": frame_index,
                            "objects": records,
                        }
                    ),
                    encoding="utf-8",
                )
                if frame_index == 0 or (frame_index + 1) % 100 == 0:
                    print(
                        f"{sequence}: {frame_index + 1}/{len(image_paths)}",
                        flush=True,
                    )
        finally:
            writer.release()
        print(f"Saved {video_path}", flush=True)


if __name__ == "__main__":
    main()
