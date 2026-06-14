from __future__ import annotations

import csv
import json
import shutil
import subprocess
import uuid
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
PART_B = ROOT / "part_B"
OUTPUT_ROOT = PART_B / "demo_outputs"
PART_MODEL_PATH = ROOT / "part_A" / "models" / "yolo" / "yolo26s_seg_best.pt"
PAPER_MODEL_PATH = PART_B / "models" / "paper" / "yolo26s_seg_paper_best.pt"

YOLO_IMAGE_SIZE = 960
PART_CONFIDENCE = 0.10
PAPER_CONFIDENCE = 0.20
REFERENCE_WIDTH = 1280.0
GAZE_TOLERANCE_PX = 30.0
MIN_STABLE_FRAMES = 3
TRANSITION_GAP_FRAMES = 2
BATCH_SIZE = 8

STATE_COLORS = {
    "instruction": (50, 190, 255),
    "assembly": (70, 220, 90),
    "elsewhere": (80, 80, 255),
    "invalid": (150, 150, 150),
}
PAPER_COLOR = (50, 190, 255)
ASSEMBLY_COLOR = (70, 220, 90)

_PART_MODEL: YOLO | None = None
_PAPER_MODEL: YOLO | None = None


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def transcode_h264(raw_path: Path, final_path: Path, logs: list[str]) -> Path:
    try:
        import imageio_ffmpeg

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
        raw_path.unlink(missing_ok=True)
        return final_path
    except Exception as error:
        logs.append(f"H.264 conversion unavailable: {error}")
        return raw_path


def load_models(logs: list[str]) -> tuple[YOLO, YOLO]:
    global _PART_MODEL, _PAPER_MODEL
    if not PART_MODEL_PATH.exists():
        raise gr.Error(f"LEGO-part model is missing: {PART_MODEL_PATH}")
    if not PAPER_MODEL_PATH.exists():
        raise gr.Error(f"Instruction-paper model is missing: {PAPER_MODEL_PATH}")
    if _PART_MODEL is None:
        logs.append("Loading LEGO-part segmentation model...")
        _PART_MODEL = YOLO(str(PART_MODEL_PATH))
    if _PAPER_MODEL is None:
        logs.append("Loading instruction-paper segmentation model...")
        _PAPER_MODEL = YOLO(str(PAPER_MODEL_PATH))
    return _PART_MODEL, _PAPER_MODEL


def parse_gaze_csv(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            try:
                if len(values) >= 3:
                    frame_name = values[0].strip()
                    x = float(values[1])
                    y = float(values[2])
                elif len(values) == 2:
                    frame_name = f"{len(rows):06d}"
                    x = float(values[0])
                    y = float(values[1])
                else:
                    raise ValueError("expected two or three columns")
            except ValueError:
                if line_number == 1:
                    continue
                raise gr.Error(
                    f"Could not parse gaze CSV line {line_number}: {values}"
                )
            rows.append({"frame_name": frame_name, "x": x, "y": y})
    if not rows:
        raise gr.Error("The gaze CSV contains no coordinate rows.")
    return rows


def result_polygons(result) -> list[np.ndarray]:
    if result.masks is None:
        return []
    return [
        np.asarray(polygon, dtype=np.float32)
        for polygon in result.masks.xy
        if len(polygon) >= 3
    ]


def point_distance(point: tuple[float, float], polygons: list[np.ndarray]) -> float:
    if not polygons:
        return float("inf")
    distances = []
    for polygon in polygons:
        signed_distance = cv2.pointPolygonTest(
            polygon,
            (float(point[0]), float(point[1])),
            True,
        )
        distances.append(0.0 if signed_distance >= 0 else -float(signed_distance))
    return min(distances)


def classify_gaze(
    gaze_x: float,
    gaze_y: float,
    width: int,
    height: int,
    paper_polygons: list[np.ndarray],
    assembly_polygons: list[np.ndarray],
) -> tuple[str, float, float]:
    valid = (
        not (gaze_x == 0 and gaze_y == 0)
        and 0 <= gaze_x < width
        and 0 <= gaze_y < height
    )
    if not valid:
        return "invalid", float("nan"), float("nan")

    paper_distance = point_distance((gaze_x, gaze_y), paper_polygons)
    assembly_distance = point_distance((gaze_x, gaze_y), assembly_polygons)
    tolerance = GAZE_TOLERANCE_PX * width / REFERENCE_WIDTH
    paper_near = paper_distance <= tolerance
    assembly_near = assembly_distance <= tolerance
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


def state_runs(states: list[str]) -> list[tuple[int, int, str]]:
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
        runs = state_runs(filtered)
        for index, (start, end, state) in enumerate(runs):
            if state == "invalid" or end - start >= MIN_STABLE_FRAMES:
                continue
            previous = runs[index - 1] if index > 0 else None
            following = runs[index + 1] if index + 1 < len(runs) else None
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


def attention_transitions(states: list[str], fps: float) -> list[dict]:
    runs = state_runs(states)
    transitions = []
    for index, first in enumerate(runs[:-1]):
        if first[2] not in {"instruction", "assembly"}:
            continue
        second = runs[index + 1]
        target = second if second[2] in {"instruction", "assembly"} else None
        if (
            second[2] in {"elsewhere", "invalid"}
            and second[1] - second[0] <= TRANSITION_GAP_FRAMES
            and index + 2 < len(runs)
        ):
            candidate = runs[index + 2]
            if candidate[2] in {"instruction", "assembly"}:
                target = candidate
        if target is not None and first[2] != target[2]:
            transitions.append(
                {
                    "from": first[2],
                    "to": target[2],
                    "frame": target[0],
                    "time_seconds": target[0] / fps,
                }
            )
    return transitions


def transition_counter(transitions: list[dict]) -> Counter:
    counts: Counter = Counter()
    for transition in transitions:
        counts[f"{transition['from']}_to_{transition['to']}"] += 1
    return counts


def draw_polygons(
    frame: np.ndarray,
    polygons: list[np.ndarray],
    color: tuple[int, int, int],
) -> None:
    integer_polygons = [np.rint(item).astype(np.int32) for item in polygons]
    if not integer_polygons:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, integer_polygons, color)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.polylines(frame, integer_polygons, True, color, 2, cv2.LINE_AA)


def draw_gaze(
    frame: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    state: str,
) -> None:
    if state == "invalid":
        return
    point = (int(round(gaze_x)), int(round(gaze_y)))
    cv2.circle(frame, point, 13, (255, 255, 255), 3)
    cv2.circle(frame, point, 8, STATE_COLORS[state], -1)


def percentages(counts: Counter) -> dict[str, float]:
    valid = counts["instruction"] + counts["assembly"] + counts["elsewhere"]
    return {
        state: 100.0 * counts[state] / valid if valid else 0.0
        for state in ("instruction", "assembly", "elsewhere")
    }


def metrics_markdown(
    processed: int,
    total: int,
    counts: Counter,
    transitions: int,
) -> str:
    values = percentages(counts)
    valid = counts["instruction"] + counts["assembly"] + counts["elsewhere"]
    return (
        f"### Live attention\n"
        f"- **Processed:** {processed}/{total} frames\n"
        f"- **Instruction:** {values['instruction']:.1f}%\n"
        f"- **Assembly:** {values['assembly']:.1f}%\n"
        f"- **Elsewhere:** {values['elsewhere']:.1f}%\n"
        f"- **Valid gaze frames:** {valid}\n"
        f"- **Instruction/assembly transitions:** {transitions}\n\n"
        "Live values are provisional until short gaze runs are filtered."
    )


def draw_metrics_panel(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    state: str,
    counts: Counter,
    transition_counts: Counter,
) -> None:
    height, width = frame.shape[:2]
    values = percentages(counts)
    valid = counts["instruction"] + counts["assembly"] + counts["elsewhere"]
    trve = (
        100.0 * (counts["instruction"] + counts["assembly"]) / valid
        if valid
        else 0.0
    )
    panel_width = min(width, 650)
    panel_height = min(height, 190)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    lines = [
        (
            f"Time {format_time(frame_index / fps)} | "
            f"Gaze: {state.upper()}",
            STATE_COLORS[state],
        ),
        (
            f"Instruction {values['instruction']:.1f}% | "
            f"Assembly {values['assembly']:.1f}% | "
            f"Elsewhere {values['elsewhere']:.1f}%",
            (245, 245, 245),
        ),
        (
            f"Transitions: {sum(transition_counts.values())} | "
            f"I->A {transition_counts['instruction_to_assembly']} | "
            f"A->I {transition_counts['assembly_to_instruction']}",
            (245, 245, 245),
        ),
        (f"Current TRVE: {trve:.1f}% of valid gaze", (210, 180, 255)),
    ]
    y = 36
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 42


def empty_transition_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Number", "From", "To", "Frame", "Timestamp"]
    )


def analyze_video(video_path: str | None, gaze_path: str | None):
    if not video_path:
        raise gr.Error("Upload a video before pressing Run analysis.")
    if not gaze_path:
        raise gr.Error("Upload the matching gaze.csv file.")

    logs = [
        "Starting Part B gaze-attention analysis.",
        "One gaze row is aligned to one uploaded video frame.",
    ]
    empty_table = empty_transition_table()
    yield None, None, "Loading models...", empty_table, "Processing...", "\n".join(logs), None

    part_model, paper_model = load_models(logs)
    gaze_rows = parse_gaze_csv(gaze_path)
    source = Path(video_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise gr.Error(f"Could not open video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 10.0
        logs.append("Video FPS was unavailable; assuming 10 FPS.")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(video_frames, len(gaze_rows)) if video_frames > 0 else len(gaze_rows)
    if video_frames != len(gaze_rows):
        logs.append(
            f"Frame-count mismatch: video={video_frames}, gaze rows={len(gaze_rows)}. "
            f"Analyzing the first {frame_limit} aligned frames."
        )
    else:
        logs.append(f"Aligned {frame_limit} video frames and gaze rows.")

    run_dir = OUTPUT_ROOT / uuid.uuid4().hex[:12]
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    raw_states: list[str] = []
    frame_rows: list[dict] = []
    provisional_counts: Counter = Counter()
    device = 0 if torch.cuda.is_available() else "cpu"

    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def process_batch() -> np.ndarray | None:
        if not batch_frames:
            return None
        part_results = part_model.predict(
            source=batch_frames,
            imgsz=YOLO_IMAGE_SIZE,
            conf=PART_CONFIDENCE,
            device=device,
            verbose=False,
            stream=False,
        )
        paper_results = paper_model.predict(
            source=batch_frames,
            imgsz=YOLO_IMAGE_SIZE,
            conf=PAPER_CONFIDENCE,
            device=device,
            verbose=False,
            stream=False,
        )
        latest_preview = None
        for frame, frame_index, part_result, paper_result in zip(
            batch_frames,
            batch_indices,
            part_results,
            paper_results,
        ):
            height, width = frame.shape[:2]
            assembly_polygons = result_polygons(part_result)
            paper_polygons = result_polygons(paper_result)
            gaze = gaze_rows[frame_index]
            state, paper_distance, assembly_distance = classify_gaze(
                gaze["x"],
                gaze["y"],
                width,
                height,
                paper_polygons,
                assembly_polygons,
            )
            raw_states.append(state)
            provisional_counts[state] += 1

            annotated = frame.copy()
            draw_polygons(annotated, paper_polygons, PAPER_COLOR)
            draw_polygons(annotated, assembly_polygons, ASSEMBLY_COLOR)
            cv2.imwrite(
                str(frame_dir / f"{frame_index:06d}.jpg"),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            frame_rows.append(
                {
                    "frame": frame_index,
                    "time_seconds": frame_index / fps,
                    "gaze_x": gaze["x"],
                    "gaze_y": gaze["y"],
                    "paper_distance_pixels": paper_distance,
                    "assembly_distance_pixels": assembly_distance,
                    "raw_state": state,
                    "paper_present": int(bool(paper_polygons)),
                    "assembly_object_count": len(assembly_polygons),
                }
            )
            preview = annotated.copy()
            draw_gaze(preview, gaze["x"], gaze["y"], state)
            provisional_transitions = attention_transitions(raw_states, fps)
            draw_metrics_panel(
                preview,
                frame_index,
                fps,
                state,
                provisional_counts,
                transition_counter(provisional_transitions),
            )
            latest_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        batch_frames.clear()
        batch_indices.clear()
        return latest_preview

    latest_preview = None
    frame_index = 0
    try:
        while frame_index < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_index)
            frame_index += 1
            if len(batch_frames) >= BATCH_SIZE:
                latest_preview = process_batch()
                if frame_index == BATCH_SIZE or frame_index % 40 == 0:
                    provisional_transition_count = len(
                        attention_transitions(raw_states, fps)
                    )
                    logs.append(f"Segmented and classified {frame_index} frames.")
                    yield (
                        None,
                        latest_preview,
                        metrics_markdown(
                            frame_index,
                            frame_limit,
                            provisional_counts,
                            provisional_transition_count,
                        ),
                        empty_table,
                        "Processing video...",
                        "\n".join(logs[-30:]),
                        None,
                    )
        batch_preview = process_batch()
        if batch_preview is not None:
            latest_preview = batch_preview
    finally:
        capture.release()

    if not frame_rows:
        raise gr.Error("No aligned frames could be analyzed.")

    filtered_states = stabilize_states(raw_states)
    transitions = attention_transitions(filtered_states, fps)
    for row, state in zip(frame_rows, filtered_states):
        row["filtered_state"] = state

    first_frame = cv2.imread(str(frame_dir / "000000.jpg"))
    height, width = first_frame.shape[:2]
    raw_video = run_dir / "attention_review_mp4v.mp4"
    final_video = run_dir / "attention_review.mp4"
    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise gr.Error("Could not create the annotated output video.")

    transitions_by_frame: dict[int, list[dict]] = {}
    for transition in transitions:
        transitions_by_frame.setdefault(transition["frame"], []).append(transition)

    final_counts: Counter = Counter()
    transition_counts: Counter = Counter()
    for row in frame_rows:
        index = int(row["frame"])
        state = row["filtered_state"]
        final_counts[state] += 1
        for transition in transitions_by_frame.get(index, []):
            transition_counts[f"{transition['from']}_to_{transition['to']}"] += 1
        frame = cv2.imread(str(frame_dir / f"{index:06d}.jpg"))
        draw_gaze(frame, row["gaze_x"], row["gaze_y"], state)
        draw_metrics_panel(
            frame,
            index,
            fps,
            state,
            final_counts,
            transition_counts,
        )
        writer.write(frame)
    writer.release()

    playable_video = transcode_h264(raw_video, final_video, logs)
    values = percentages(final_counts)
    valid_frames = (
        final_counts["instruction"]
        + final_counts["assembly"]
        + final_counts["elsewhere"]
    )
    relevant_frames = final_counts["instruction"] + final_counts["assembly"]
    trve = 100.0 * relevant_frames / valid_frames if valid_frames else 0.0
    valid_seconds = valid_frames / fps

    transition_table = pd.DataFrame(
        [
            [
                number,
                item["from"].title(),
                item["to"].title(),
                item["frame"],
                format_time(item["time_seconds"]),
            ]
            for number, item in enumerate(transitions, start=1)
        ],
        columns=empty_table.columns,
    )

    csv_path = run_dir / "attention_by_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    summary_data = {
        "input_video": str(source),
        "input_gaze_csv": str(gaze_path),
        "fps": fps,
        "analyzed_frames": len(frame_rows),
        "duration_seconds": len(frame_rows) / fps,
        "valid_gaze_frames": valid_frames,
        "valid_gaze_seconds": valid_seconds,
        "valid_time_percent": values,
        "instruction_to_assembly_transitions": transition_counts[
            "instruction_to_assembly"
        ],
        "assembly_to_instruction_transitions": transition_counts[
            "assembly_to_instruction"
        ],
        "total_instruction_assembly_transitions": len(transitions),
        "task_relevant_visual_engagement_percent": trve,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_data, indent=2),
        encoding="utf-8",
    )
    shutil.rmtree(frame_dir, ignore_errors=True)

    final_summary = (
        f"## Final result\n"
        f"- **Task-Relevant Visual Engagement (TRVE): {trve:.1f}%**\n"
        f"- **Instruction:** {values['instruction']:.1f}% of valid gaze "
        f"({final_counts['instruction'] / fps:.1f} s)\n"
        f"- **Assembly:** {values['assembly']:.1f}% of valid gaze "
        f"({final_counts['assembly'] / fps:.1f} s)\n"
        f"- **Elsewhere:** {values['elsewhere']:.1f}% of valid gaze "
        f"({final_counts['elsewhere'] / fps:.1f} s)\n"
        f"- **Invalid gaze:** {final_counts['invalid'] / fps:.1f} s\n"
        f"- **Instruction/assembly transitions:** {len(transitions)}\n"
        f"- **Analyzed duration:** {len(frame_rows) / fps:.1f} s\n\n"
        "TRVE is the percentage of valid gaze time directed to either the "
        "instruction paper or the LEGO assembly."
    )
    logs.append(f"Analysis complete. Final TRVE: {trve:.1f}%.")
    logs.append(f"Annotated video saved to {playable_video}.")
    yield (
        str(playable_video),
        latest_preview,
        metrics_markdown(
            len(frame_rows),
            len(frame_rows),
            final_counts,
            len(transitions),
        ).replace("Live attention", "Final attention"),
        transition_table,
        final_summary,
        "\n".join(logs[-50:]),
        str(csv_path),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Part B Gaze-Attention Demo") as app:
        gr.Markdown(
            "# Part B Gaze-Attention Video Demo\n"
            "Upload a construction video and its frame-aligned `gaze.csv`, then "
            "press **Run analysis**. The app segments the instruction paper and "
            "LEGO parts, visualizes gaze, tracks cumulative attention percentages "
            "and transitions, and reports final TRVE."
        )
        with gr.Row():
            video_input = gr.Video(
                label="Input construction video",
                sources=["upload"],
            )
            gaze_input = gr.File(
                label="Matching gaze.csv",
                file_types=[".csv"],
                type="filepath",
            )
        run_button = gr.Button("Run analysis", variant="primary")
        with gr.Row():
            preview = gr.Image(
                label="Live processing preview",
                type="numpy",
            )
            live_metrics = gr.Markdown("Upload both files to begin.")
        final_summary = gr.Markdown()
        output_video = gr.Video(label="Annotated gaze-attention video")
        transition_table = gr.Dataframe(
            headers=["Number", "From", "To", "Frame", "Timestamp"],
            datatype=["number", "str", "str", "number", "str"],
            interactive=False,
            label="Instruction/assembly transitions",
        )
        logs = gr.Textbox(
            label="Processing log",
            lines=14,
            interactive=False,
        )
        frame_csv = gr.File(label="Frame-level attention results")
        run_button.click(
            fn=analyze_video,
            inputs=[video_input, gaze_input],
            outputs=[
                output_video,
                preview,
                live_metrics,
                transition_table,
                final_summary,
                logs,
                frame_csv,
            ],
        )
    return app


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
    )
