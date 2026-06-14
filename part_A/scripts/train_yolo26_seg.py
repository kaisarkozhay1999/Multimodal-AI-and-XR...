from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PSR_DATA_ROOT", ROOT)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26 segmentation.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_ROOT / "samurai_review" / "yolo10_dataset" / "data.yaml",
    )
    parser.add_argument("--model", default="yolo26s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo26s_seg_10class")
    parser.add_argument("--project", type=Path, default=ROOT / "models" / "yolo_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        pretrained=True,
        patience=20,
        save=True,
        save_period=10,
        cache=False,
        plots=True,
        deterministic=True,
        seed=42,
        close_mosaic=10,
        amp=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
