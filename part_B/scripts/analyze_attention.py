from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PART_B = ROOT / "part_B"
SEQUENCE = "22_assy_0_1"
SESSION = "22_assy_0_1_paper_000000_000199"
FPS = 10.0
FRAME_COUNT = 200
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
GAZE_TOLERANCE_PX = 30.0
MIN_STABLE_FRAMES = 3
TRANSITION_GAP_FRAMES = 2

PAPER_DIR = (
    PART_B / "paper_tracking" / "annotations" / SESSION / "accepted"
)
ASSEMBLY_DIR = ROOT / "samurai_review" / "data" / SEQUENCE / "accepted"
FRAME_DIR = ROOT / "train" / SEQUENCE / "rgb"
GAZE_PATH = ROOT / "train" / SEQUENCE / "gaze.csv"
AR_LABELS_PATH = ROOT / "train" / SEQUENCE / "AR_labels.csv"
OUTPUT_DIR = PART_B / "results" / "attention_200_frames"

STATE_COLORS_BGR = {
    "instruction": (50, 190, 255),
    "assembly": (70, 220, 90),
    "elsewhere": (80, 80, 255),
    "invalid": (150, 150, 150),
}
STATE_COLORS_PLOT = {
    "instruction": "#f9a825",
    "assembly": "#2e7d32",
    "elsewhere": "#d32f2f",
    "invalid": "#8d8d8d",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def polygon_array(obj: dict) -> np.ndarray | None:
    polygon = np.asarray(obj.get("polygon", []), dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return None
    return polygon


def point_distance(point: tuple[int, int], polygons: list[np.ndarray]) -> float:
    if not polygons:
        return float("inf")
    distances = []
    for polygon in polygons:
        signed = cv2.pointPolygonTest(
            polygon,
            (float(point[0]), float(point[1])),
            True,
        )
        distances.append(0.0 if signed >= 0 else -float(signed))
    return min(distances)


def load_gaze() -> list[tuple[str, int, int]]:
    rows = []
    with GAZE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for frame_name, x, y in csv.reader(handle):
            rows.append((frame_name, int(x), int(y)))
    if len(rows) < FRAME_COUNT:
        raise ValueError(f"Only {len(rows)} gaze rows found")
    return rows[:FRAME_COUNT]


def load_polygons(path: Path) -> list[np.ndarray]:
    data = load_json(path)
    polygons = []
    for obj in data.get("objects", []):
        polygon = polygon_array(obj)
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def classify_frame(
    gaze_x: int,
    gaze_y: int,
    paper_polygons: list[np.ndarray],
    assembly_polygons: list[np.ndarray],
) -> tuple[str, float, float]:
    valid = (
        not (gaze_x == 0 and gaze_y == 0)
        and 0 <= gaze_x < IMAGE_WIDTH
        and 0 <= gaze_y < IMAGE_HEIGHT
    )
    if not valid:
        return "invalid", float("nan"), float("nan")

    point = (gaze_x, gaze_y)
    paper_distance = point_distance(point, paper_polygons)
    assembly_distance = point_distance(point, assembly_polygons)
    paper_near = paper_distance <= GAZE_TOLERANCE_PX
    assembly_near = assembly_distance <= GAZE_TOLERANCE_PX
    if paper_near and assembly_near:
        state = (
            "instruction"
            if paper_distance <= assembly_distance
            else "assembly"
        )
    elif paper_near:
        state = "instruction"
    elif assembly_near:
        state = "assembly"
    else:
        state = "elsewhere"
    return state, paper_distance, assembly_distance


def runs(states: list[str]) -> list[tuple[int, int, str]]:
    if not states:
        return []
    output = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            output.append((start, index, states[start]))
            start = index
    return output


def stabilize_states(states: list[str]) -> list[str]:
    filtered = list(states)
    changed = True
    while changed:
        changed = False
        current_runs = runs(filtered)
        for run_index, (start, end, state) in enumerate(current_runs):
            if state == "invalid" or end - start >= MIN_STABLE_FRAMES:
                continue
            previous = current_runs[run_index - 1] if run_index > 0 else None
            following = (
                current_runs[run_index + 1]
                if run_index + 1 < len(current_runs)
                else None
            )
            replacement = None
            if (
                previous
                and following
                and previous[2] == following[2]
                and previous[2] != "invalid"
            ):
                replacement = previous[2]
            elif previous and previous[2] != "invalid":
                replacement = previous[2]
            elif following and following[2] != "invalid":
                replacement = following[2]
            if replacement is not None:
                filtered[start:end] = [replacement] * (end - start)
                changed = True
                break
    return filtered


def attention_transitions(states: list[str]) -> list[dict]:
    state_runs = runs(states)
    transitions = []
    for index, first in enumerate(state_runs[:-1]):
        if first[2] not in {"instruction", "assembly"}:
            continue
        second = state_runs[index + 1]
        target = None
        if second[2] in {"instruction", "assembly"}:
            target = second
        if (
            second[2] in {"elsewhere", "invalid"}
            and second[1] - second[0] <= TRANSITION_GAP_FRAMES
            and index + 2 < len(state_runs)
        ):
            candidate = state_runs[index + 2]
            if candidate[2] in {"instruction", "assembly"}:
                target = candidate
        if (
            target is not None
            and first[2] != target[2]
        ):
            transitions.append(
                {
                    "from": first[2],
                    "to": target[2],
                    "frame": target[0],
                    "time_seconds": target[0] / FPS,
                }
            )
    return transitions


def instruction_intervals() -> list[tuple[int, int]]:
    intervals = []
    with AR_LABELS_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 5 or row[2] != "check_instruction":
                continue
            start = int(Path(row[3]).stem)
            end = int(Path(row[4]).stem)
            if start >= FRAME_COUNT:
                continue
            intervals.append((start, min(end, FRAME_COUNT - 1)))
    return intervals


def summarize(states: list[str], transitions: list[dict]) -> dict:
    counts = Counter(states)
    valid_frames = len(states) - counts["invalid"]
    valid_seconds = valid_frames / FPS
    summary = {
        "sequence": SEQUENCE,
        "session": SESSION,
        "fps": FPS,
        "frame_count": len(states),
        "duration_seconds": len(states) / FPS,
        "gaze_tolerance_pixels": GAZE_TOLERANCE_PX,
        "minimum_stable_frames": MIN_STABLE_FRAMES,
        "transition_gap_frames": TRANSITION_GAP_FRAMES,
        "valid_gaze_frames": valid_frames,
        "invalid_gaze_frames": counts["invalid"],
        "valid_gaze_seconds": valid_seconds,
        "state_frames": dict(counts),
        "state_seconds": {
            state: counts[state] / FPS
            for state in ("instruction", "assembly", "elsewhere", "invalid")
        },
        "valid_time_percent": {
            state: (
                100.0 * counts[state] / valid_frames if valid_frames else 0.0
            )
            for state in ("instruction", "assembly", "elsewhere")
        },
        "instruction_to_assembly_transitions": sum(
            item["from"] == "instruction" and item["to"] == "assembly"
            for item in transitions
        ),
        "assembly_to_instruction_transitions": sum(
            item["from"] == "assembly" and item["to"] == "instruction"
            for item in transitions
        ),
        "bidirectional_transitions": len(transitions),
        "transitions_per_valid_minute": (
            len(transitions) / (valid_seconds / 60.0)
            if valid_seconds > 0
            else 0.0
        ),
    }

    intervals = instruction_intervals()
    interval_frames = sorted(
        {
            frame
            for start, end in intervals
            for frame in range(start, end + 1)
        }
    )
    valid_interval_frames = [
        frame for frame in interval_frames if states[frame] != "invalid"
    ]
    instruction_hits = sum(
        states[frame] == "instruction" for frame in valid_interval_frames
    )
    summary["check_instruction_validation"] = {
        "intervals": [[start, end] for start, end in intervals],
        "valid_frames": len(valid_interval_frames),
        "instruction_classified_frames": instruction_hits,
        "instruction_agreement_percent": (
            100.0 * instruction_hits / len(valid_interval_frames)
            if valid_interval_frames
            else 0.0
        ),
    }
    return summary


def plot_timeline(rows: list[dict], summary: dict) -> None:
    state_values = {
        "invalid": 0,
        "elsewhere": 1,
        "assembly": 2,
        "instruction": 3,
    }
    values = np.asarray(
        [state_values[row["filtered_state"]] for row in rows],
        dtype=np.float32,
    )
    times = np.arange(len(rows), dtype=np.float32) / FPS

    figure, (axis, bar_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 6.5),
        gridspec_kw={"height_ratios": [2.3, 1]},
        constrained_layout=True,
    )
    axis.step(times, values, where="post", color="#263238", linewidth=1.4)
    for state, value in state_values.items():
        mask = values == value
        axis.scatter(
            times[mask],
            values[mask],
            s=16,
            color=STATE_COLORS_PLOT[state],
            label=state.title(),
            zorder=3,
        )
    for start, end in instruction_intervals():
        axis.axvspan(start / FPS, (end + 1) / FPS, color="#42a5f5", alpha=0.12)
    axis.set_yticks(list(state_values.values()))
    axis.set_yticklabels([item.title() for item in state_values])
    axis.set_ylim(-0.35, 3.35)
    axis.set_xlabel("Time (seconds)")
    axis.set_title("Gaze allocation over the 200-frame paper-tracking interval")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper right", ncol=4)

    states = ["instruction", "assembly", "elsewhere"]
    percentages = [
        summary["valid_time_percent"][state] for state in states
    ]
    bars = bar_axis.bar(
        [state.title() for state in states],
        percentages,
        color=[STATE_COLORS_PLOT[state] for state in states],
    )
    for bar, value in zip(bars, percentages):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    bar_axis.set_ylim(0, max(100, max(percentages, default=0) + 10))
    bar_axis.set_ylabel("Percent of valid gaze time")
    bar_axis.grid(axis="y", alpha=0.2)
    figure.savefig(OUTPUT_DIR / "attention_timeline.png", dpi=180)
    plt.close(figure)


def draw_polygons(
    frame: np.ndarray,
    polygons: list[np.ndarray],
    color: tuple[int, int, int],
) -> None:
    overlay = frame.copy()
    integer_polygons = [np.rint(item).astype(np.int32) for item in polygons]
    if integer_polygons:
        cv2.fillPoly(overlay, integer_polygons, color)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.polylines(frame, integer_polygons, True, color, 2, cv2.LINE_AA)


def render_review_video(rows: list[dict]) -> None:
    raw_path = OUTPUT_DIR / "attention_review_mp4v.mp4"
    final_path = OUTPUT_DIR / "attention_review.mp4"
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create review video")

    for row in rows:
        frame_index = int(row["frame"])
        frame = cv2.imread(str(FRAME_DIR / f"{frame_index:06d}.jpg"))
        paper = load_polygons(PAPER_DIR / f"{frame_index:06d}.json")
        assembly = load_polygons(ASSEMBLY_DIR / f"{frame_index:06d}.json")
        draw_polygons(frame, paper, STATE_COLORS_BGR["instruction"])
        draw_polygons(frame, assembly, STATE_COLORS_BGR["assembly"])

        state = row["filtered_state"]
        gaze_x, gaze_y = int(row["gaze_x"]), int(row["gaze_y"])
        if state != "invalid":
            color = STATE_COLORS_BGR[state]
            cv2.circle(frame, (gaze_x, gaze_y), 12, (255, 255, 255), 3)
            cv2.circle(frame, (gaze_x, gaze_y), 8, color, -1)

        cv2.rectangle(frame, (0, 0), (590, 58), (15, 15, 15), -1)
        cv2.putText(
            frame,
            f"Frame {frame_index:03d} | {frame_index / FPS:05.1f}s | "
            f"Gaze: {state.upper()}",
            (14, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            STATE_COLORS_BGR[state],
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()

    try:
        import imageio_ffmpeg
        import subprocess

        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final_path),
            ],
            check=True,
        )
        raw_path.unlink()
    except Exception:
        raw_path.replace(final_path)


def write_report(summary: dict) -> None:
    percentages = summary["valid_time_percent"]
    validation = summary["check_instruction_validation"]
    report = f"""# Initial Gaze-Attention Results

This prototype covers source frames `000000`-`000199` of `{SEQUENCE}`
(`20.0` seconds at 10 FPS).

## Classification

- Instruction paper: gaze lies inside or within {GAZE_TOLERANCE_PX:.0f} pixels of the accepted paper mask.
- Assembly: gaze lies inside or within {GAZE_TOLERANCE_PX:.0f} pixels of any accepted LEGO-part mask.
- Elsewhere: valid gaze lies near neither target.
- Invalid: `(0, 0)` or outside the 1280 x 720 image.
- Short state runs below {MIN_STABLE_FRAMES} frames are filtered before reporting.
- Transitions may bridge a gap of at most {TRANSITION_GAP_FRAMES / FPS:.1f} seconds; longer invalid periods break the transition chain.

## Results

- Paper visible: {summary["target_visibility"]["paper_visible_frames"]} / {summary["frame_count"]} frames
- At least one assembly mask visible: {summary["target_visibility"]["assembly_visible_frames"]} / {summary["frame_count"]} frames
- Instruction: {summary["state_seconds"]["instruction"]:.1f} s ({percentages["instruction"]:.1f}% of valid gaze)
- Assembly: {summary["state_seconds"]["assembly"]:.1f} s ({percentages["assembly"]:.1f}% of valid gaze)
- Elsewhere: {summary["state_seconds"]["elsewhere"]:.1f} s ({percentages["elsewhere"]:.1f}% of valid gaze)
- Invalid gaze: {summary["state_seconds"]["invalid"]:.1f} s
- Instruction to assembly transitions: {summary["instruction_to_assembly_transitions"]}
- Assembly to instruction transitions: {summary["assembly_to_instruction_transitions"]}
- Total bidirectional transitions: {summary["bidirectional_transitions"]}
- Transition rate: {summary["transitions_per_valid_minute"]:.2f} per valid minute

## Sanity Check

During the `check_instruction` intervals from `AR_labels.csv`,
{validation["instruction_agreement_percent"]:.1f}% of valid gaze frames were
classified as instruction-paper attention.

This is an initial 20-second prototype. It should be visually checked using
`attention_review.mp4` before the method is applied to longer intervals.
"""
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper_files = sorted(PAPER_DIR.glob("*.json"))
    assembly_files = sorted(ASSEMBLY_DIR.glob("*.json"))
    if len(paper_files) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} paper masks, found {len(paper_files)}")
    if len(assembly_files) < FRAME_COUNT:
        raise ValueError(
            f"Expected at least {FRAME_COUNT} assembly masks, found {len(assembly_files)}"
        )

    gaze_rows = load_gaze()
    rows = []
    raw_states = []
    for frame_index, (frame_name, gaze_x, gaze_y) in enumerate(gaze_rows):
        if int(Path(frame_name).stem) != frame_index:
            raise ValueError(f"Unexpected gaze frame order at {frame_name}")
        paper = load_polygons(PAPER_DIR / f"{frame_index:06d}.json")
        assembly = load_polygons(ASSEMBLY_DIR / f"{frame_index:06d}.json")
        state, paper_distance, assembly_distance = classify_frame(
            gaze_x,
            gaze_y,
            paper,
            assembly,
        )
        raw_states.append(state)
        rows.append(
            {
                "frame": frame_index,
                "time_seconds": frame_index / FPS,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "paper_distance_pixels": paper_distance,
                "assembly_distance_pixels": assembly_distance,
                "raw_state": state,
                "paper_present": int(bool(paper)),
                "assembly_object_count": len(assembly),
            }
        )

    filtered_states = stabilize_states(raw_states)
    for row, state in zip(rows, filtered_states):
        row["filtered_state"] = state

    transitions = attention_transitions(filtered_states)
    summary = summarize(filtered_states, transitions)
    summary["raw_state_frames"] = dict(Counter(raw_states))
    summary["target_visibility"] = {
        "paper_visible_frames": sum(row["paper_present"] for row in rows),
        "paper_absent_frames": sum(not row["paper_present"] for row in rows),
        "assembly_visible_frames": sum(
            row["assembly_object_count"] > 0 for row in rows
        ),
    }

    with (OUTPUT_DIR / "attention_by_frame.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "transitions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["from", "to", "frame", "time_seconds"],
        )
        writer.writeheader()
        writer.writerows(transitions)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_timeline(rows, summary)
    render_review_video(rows)
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
