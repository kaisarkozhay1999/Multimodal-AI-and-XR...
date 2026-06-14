from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PART_B_ROOT = ROOT / "part_B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a consecutive-frame SAMURAI paper-tracking session."
    )
    parser.add_argument("--sequence", default="22_assy_0_1")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=200)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.frame_count <= 0:
        raise ValueError("start-frame must be non-negative and frame-count positive")

    source_rgb = ROOT / "train" / args.sequence / "rgb"
    if not source_rgb.is_dir():
        raise FileNotFoundError(f"Missing source RGB directory: {source_rgb}")

    end_frame = args.start_frame + args.frame_count - 1
    session_name = (
        f"{args.sequence}_paper_{args.start_frame:06d}_{end_frame:06d}"
    )
    destination_rgb = (
        PART_B_ROOT / "paper_tracking" / "sequences" / session_name / "rgb"
    )
    destination_rgb.mkdir(parents=True, exist_ok=True)

    counts = {"linked": 0, "copied": 0, "existing": 0}
    source_frames = []
    for local_index, source_index in enumerate(
        range(args.start_frame, end_frame + 1)
    ):
        source = source_rgb / f"{source_index:06d}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Missing source frame: {source}")
        destination = destination_rgb / f"{local_index:06d}.jpg"
        counts[link_or_copy(source, destination)] += 1
        source_frames.append(
            {
                "session_frame": local_index,
                "source_frame": source_index,
                "source_name": source.name,
            }
        )

    manifest = {
        "version": 1,
        "session": session_name,
        "source_sequence": args.sequence,
        "source_start_frame": args.start_frame,
        "source_end_frame": end_frame,
        "frame_count": args.frame_count,
        "class": "instruction_paper",
        "frame_mapping": source_frames,
    }
    manifest_path = destination_rgb.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "session": session_name,
                "destination": str(destination_rgb),
                **counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
