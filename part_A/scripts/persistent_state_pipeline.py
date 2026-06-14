from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psr_metrics import (
    EVENT_LABELS,
    event_timing_metrics,
    expand_recognitions,
)
from psr_geometry import (
    EVENT_NAMES,
    ROOT,
    TEST_SEQUENCES,
    event_completion_frames,
    load_psr_rows,
    save_json,
    smooth_probabilities,
)


FPS = 10.0
TOLERANCE = 15
FIXED_CALIBRATION = {
    "selection_source": "Previously selected from four leave-one-sequence-out training predictions",
    "smooth_window": 31,
    "persistence_frames": 20,
    "hysteresis_margin": 0.15,
    "enter_thresholds": {
        "front_chassis_pair": 0.65,
        "rear_chassis_variant": 0.95,
        "front_rear_chassis_pin": 0.50,
        "rear_rear_chassis_pin": 0.95,
        "front_bracket_pair": 0.95,
        "front_wheel_assy": 0.95,
        "rear_wheel_assy": 0.65,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate the persistent state-TCN pipeline."
    )
    parser.add_argument(
        "--state-models", type=Path, default=ROOT / "models" / "state"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "final_evaluation",
    )
    return parser.parse_args()


def persistent_crossing(
    probabilities: np.ndarray,
    enter_threshold: float,
    persistence: int,
    min_index: int = 0,
    hysteresis_margin: float = 0.15,
) -> int | None:
    exit_threshold = max(0.0, enter_threshold - hysteresis_margin)
    candidates = np.flatnonzero(probabilities >= enter_threshold)
    for start in candidates:
        start = int(start)
        if start < min_index:
            continue
        end = start + persistence
        if end > len(probabilities):
            break
        if np.all(probabilities[start:end] >= exit_threshold):
            # Recognition occurs after the persistence requirement is verified.
            return end - 1
    return None


def decode_sequence(
    probabilities: np.ndarray,
    frame_numbers: np.ndarray,
    thresholds: list[float],
    smooth_window: int,
    persistence: int,
) -> tuple[dict[str, int], np.ndarray]:
    smoothed = smooth_probabilities(probabilities, smooth_window)
    indices: dict[str, int] = {}

    # Foundation events have flexible internal order.
    for event_index in range(4):
        index = persistent_crossing(
            smoothed[:, event_index],
            thresholds[event_index],
            persistence,
        )
        if index is not None:
            indices[EVENT_NAMES[event_index]] = index

    # Finishing events are accepted only after detected foundation events.
    foundation_limit = max(indices.values(), default=0)
    bracket_index = persistent_crossing(
        smoothed[:, 4], thresholds[4], persistence, foundation_limit
    )
    if bracket_index is not None:
        indices[EVENT_NAMES[4]] = bracket_index

    front_wheel_limit = max(
        foundation_limit,
        bracket_index if bracket_index is not None else foundation_limit,
    )
    front_wheel_index = persistent_crossing(
        smoothed[:, 5], thresholds[5], persistence, front_wheel_limit
    )
    if front_wheel_index is not None:
        indices[EVENT_NAMES[5]] = front_wheel_index

    rear_wheel_index = persistent_crossing(
        smoothed[:, 6], thresholds[6], persistence, foundation_limit
    )
    if rear_wheel_index is not None:
        indices[EVENT_NAMES[6]] = rear_wheel_index

    recognized = {
        event_name: int(frame_numbers[index])
        for event_name, index in indices.items()
    }
    return recognized, smoothed


def calibration_settings() -> dict:
    return {
        **FIXED_CALIBRATION,
        "enter_thresholds": dict(FIXED_CALIBRATION["enter_thresholds"]),
    }


def plot_timeline(
    path: Path,
    frame_numbers: np.ndarray,
    smoothed: np.ndarray,
    recognized: dict[str, int],
    thresholds: list[float],
) -> None:
    completion = event_completion_frames(TEST_SEQUENCES[0])
    figure, axes = plt.subplots(
        len(EVENT_NAMES),
        1,
        figsize=(16, 17),
        sharex=True,
        constrained_layout=True,
    )
    for event_index, (axis, event_name) in enumerate(zip(axes, EVENT_NAMES)):
        threshold = thresholds[event_index]
        axis.plot(
            frame_numbers,
            smoothed[:, event_index],
            color="#1565c0",
            linewidth=1.4,
            label="Smoothed state probability",
        )
        axis.axhline(
            threshold,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            label="Per-step decision threshold",
        )
        axis.text(
            0.995,
            threshold + 0.025,
            f"threshold {threshold:.2f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        true_frame = completion.get(event_name)
        if true_frame is not None:
            axis.axvspan(
                true_frame - TOLERANCE,
                true_frame + TOLERANCE,
                color="#ef5350",
                alpha=0.12,
            )
            axis.axvline(
                true_frame,
                color="#d32f2f",
                linewidth=1.5,
                label="Ground-truth completion",
            )
        predicted_frame = recognized.get(event_name)
        if predicted_frame is not None:
            axis.axvline(
                predicted_frame,
                color="#2e7d32",
                linestyle=":",
                linewidth=1.7,
                label="Persistent completion recognition",
            )
        axis.set_ylim(-0.03, 1.05)
        axis.set_ylabel("Probability")
        axis.set_title(EVENT_LABELS[event_name], loc="left", fontsize=10)
        axis.grid(axis="x", alpha=0.18)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Frame number (10 FPS)")
    figure.suptitle(
        "State-TCN pipeline: persistent completion probabilities",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_dir = args.state_models / "test_probabilities"
    calibration = calibration_settings()
    thresholds = [
        calibration["enter_thresholds"][event_name] for event_name in EVENT_NAMES
    ]

    test_sequence = TEST_SEQUENCES[0]
    test_data = np.load(test_dir / f"tcn_{test_sequence}.npz")
    recognized, smoothed = decode_sequence(
        test_data["probabilities"],
        test_data["frame_numbers"],
        thresholds,
        calibration["smooth_window"],
        calibration["persistence_frames"],
    )
    gt_rows = load_psr_rows(test_sequence)
    predictions = expand_recognitions(
        recognized, {step_id for _, step_id, _ in gt_rows}
    )
    metrics = event_timing_metrics(
        event_completion_frames(test_sequence),
        recognized,
    )

    save_json(output / "calibration.json", calibration)
    save_json(
        output / "metrics.json",
        {
            "sequence": test_sequence,
            "pipeline": "single persistent state TCN",
            "recognized_event_frames": recognized,
            "decoder": {
                "smooth_window": calibration["smooth_window"],
                "persistence_frames": calibration["persistence_frames"],
                "hysteresis_margin": calibration["hysteresis_margin"],
                "enter_thresholds": calibration["enter_thresholds"],
                "monotonic_completion": True,
                "assembly_rules": True,
            },
            "timing_metrics": metrics,
        },
    )
    with (output / "threshold_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "procedure_step",
                "psr_label_code",
                "event",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)
    np.savez_compressed(
        output / "test_state_probabilities.npz",
        frame_numbers=test_data["frame_numbers"],
        raw_probabilities=test_data["probabilities"],
        smoothed_probabilities=smoothed,
        event_names=np.asarray(EVENT_NAMES),
        thresholds=np.asarray(thresholds, dtype=np.float32),
    )
    plot_timeline(
        output / "timeline.png",
        test_data["frame_numbers"],
        smoothed,
        recognized,
        thresholds,
    )
    print(
        json.dumps(
            {
                "calibration": {
                    key: value
                    for key, value in calibration.items()
                    if key != "loso_predictions"
                },
                "recognized": recognized,
                "test_timing_metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"per_event", "pairwise_intervals"}
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
