from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
PART_B = ROOT / "part_B"
DATA_YAML = PART_B / "data" / "paper_yolo_dataset" / "data.yaml"
PROJECT_DIR = PART_B / "models" / "paper" / "runs"
MODEL_DIR = PART_B / "models" / "paper"
FINAL_MODEL = MODEL_DIR / "yolo26s_seg_paper_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one-class instruction-paper segmentation."
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset is missing: {DATA_YAML}. Run export_paper_yolo_dataset.py."
        )

    model = YOLO(str(ROOT / "yolo26s-seg.pt"))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,
        project=str(PROJECT_DIR),
        name="yolo26s_seg_instruction_paper",
        pretrained=True,
        patience=12,
        save=True,
        cache=False,
        plots=True,
        deterministic=True,
        seed=42,
        close_mosaic=8,
        amp=True,
        exist_ok=True,
    )

    best_model = (
        PROJECT_DIR
        / "yolo26s_seg_instruction_paper"
        / "weights"
        / "best.pt"
    )
    if not best_model.exists():
        raise FileNotFoundError(f"Training did not create {best_model}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, FINAL_MODEL)
    print(f"Paper model: {FINAL_MODEL}")


if __name__ == "__main__":
    main()
