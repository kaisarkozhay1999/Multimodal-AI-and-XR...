from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PSR_DATA_ROOT", ROOT / "data" / "labels")).resolve()
TRAIN_SEQUENCES = [
    "22_assy_0_1",
    "22_assy_2_3",
    "25_assy_0_1",
    "25_assy_2_1",
]
TEST_SEQUENCES = ["27_assy_0_1"]

CLASS_NAMES = [
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

# Raw PSR label codes that are observationally inseparable are grouped into one
# event. These codes map to nine human-facing procedure steps in the report.
EVENTS = [
    ("front_chassis_pair", (3, 6)),
    ("rear_chassis_variant", (9, 12)),
    ("front_rear_chassis_pin", (15,)),
    ("rear_rear_chassis_pin", (18,)),
    ("front_bracket_pair", (21, 24)),
    ("front_wheel_assy", (27,)),
    ("rear_wheel_assy", (30,)),
]
EVENT_NAMES = [name for name, _ in EVENTS]


def sequence_dir(sequence: str) -> Path:
    train = DATA_ROOT / "train" / sequence
    if train.exists():
        return train
    test = DATA_ROOT / "test" / sequence
    if test.exists():
        return test
    raise FileNotFoundError(f"Unknown sequence: {sequence}")


def load_psr_rows(sequence: str) -> list[tuple[int, int, str]]:
    path = sequence_dir(sequence) / "PSR_labels.csv"
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for frame_name, step_id, label in csv.reader(handle):
            rows.append((int(Path(frame_name).stem), int(step_id), label))
    return rows


def event_completion_frames(sequence: str) -> dict[str, int]:
    by_step = {step_id: frame for frame, step_id, _ in load_psr_rows(sequence)}
    completion = {}
    for event_name, step_ids in EVENTS:
        frames = [by_step[step_id] for step_id in step_ids if step_id in by_step]
        if frames:
            if len(set(frames)) != 1:
                raise ValueError(
                    f"{sequence}: grouped steps {step_ids} do not share a frame"
                )
            completion[event_name] = frames[0]
    return completion


def make_event_targets(
    frame_numbers: np.ndarray,
    sequence: str,
    tolerance: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    completion = event_completion_frames(sequence)
    targets = np.zeros((len(frame_numbers), len(EVENTS)), dtype=np.float32)
    present = np.zeros(len(EVENTS), dtype=bool)
    for event_index, (event_name, _) in enumerate(EVENTS):
        if event_name not in completion:
            continue
        present[event_index] = True
        targets[:, event_index] = (
            np.abs(frame_numbers - completion[event_name]) <= tolerance
        ).astype(np.float32)
    return targets, present


def make_state_targets(
    frame_numbers: np.ndarray,
    sequence: str,
    tolerance: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    completion = event_completion_frames(sequence)
    targets = np.zeros((len(frame_numbers), len(EVENTS)), dtype=np.float32)
    valid = np.ones_like(targets, dtype=np.float32)
    present = np.zeros(len(EVENTS), dtype=bool)
    for event_index, (event_name, _) in enumerate(EVENTS):
        if event_name not in completion:
            continue
        present[event_index] = True
        frame = completion[event_name]
        targets[:, event_index] = (frame_numbers > frame + tolerance).astype(
            np.float32
        )
        valid[:, event_index] = (
            np.abs(frame_numbers - frame) > tolerance
        ).astype(np.float32)
    return targets, valid, present


def temporal_tabular_features(features: np.ndarray) -> np.ndarray:
    parts = [features]
    for lag in (5, 15, 30):
        shifted = np.empty_like(features)
        shifted[:lag] = features[0]
        shifted[lag:] = features[:-lag]
        parts.append(features - shifted)
    for window in (15, 31):
        cumulative = np.vstack(
            [
                np.zeros((1, features.shape[1]), dtype=np.float64),
                np.cumsum(features, axis=0, dtype=np.float64),
            ]
        )
        ends = np.arange(1, len(features) + 1)
        starts = np.maximum(0, ends - window)
        means = (cumulative[ends] - cumulative[starts]) / (
            ends - starts
        )[:, None]
        parts.append(means.astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def smooth_probabilities(probabilities: np.ndarray, window: int = 11) -> np.ndarray:
    if window <= 1:
        return probabilities.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.column_stack(
        [
            np.convolve(probabilities[:, index], kernel, mode="same")
            for index in range(probabilities.shape[1])
        ]
    ).astype(np.float32)


def state_to_event_scores(
    state_probabilities: np.ndarray,
    transition_window: int = 15,
) -> np.ndarray:
    smoothed = smooth_probabilities(state_probabilities, window=11)
    cumulative = np.vstack(
        [
            np.zeros((1, smoothed.shape[1]), dtype=np.float64),
            np.cumsum(smoothed, axis=0, dtype=np.float64),
        ]
    )
    indices = np.arange(len(smoothed))
    before_start = np.maximum(0, indices - transition_window)
    before_count = np.maximum(1, indices - before_start)
    before = (cumulative[indices] - cumulative[before_start]) / before_count[:, None]
    after_end = np.minimum(len(smoothed), indices + transition_window + 1)
    after_count = np.maximum(1, after_end - indices)
    after = (cumulative[after_end] - cumulative[indices]) / after_count[:, None]
    transition = np.clip(after - before, 0.0, 1.0)
    crossing = np.exp(-0.5 * ((smoothed - 0.5) / 0.2) ** 2)
    return (transition + 0.05 * crossing).astype(np.float32)


def _peak_candidates(
    scores: np.ndarray,
    count: int = 50,
    separation: int = 5,
) -> np.ndarray:
    selected = []
    for index in np.argsort(scores)[::-1]:
        if all(abs(int(index) - prior) >= separation for prior in selected):
            selected.append(int(index))
        if len(selected) >= count:
            break
    selected.extend([0, len(scores) - 1])
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def decode_completion_events(
    probabilities: np.ndarray,
    frame_numbers: np.ndarray,
) -> dict[str, int]:
    """Find one event frame per class under the observed partial-order rules."""
    scores = smooth_probabilities(probabilities)
    candidates = [_peak_candidates(scores[:, index]) for index in range(len(EVENTS))]

    # E=front bracket pair, F=front wheel, G=rear wheel. E must not follow F.
    best_score = -np.inf
    best_indices = None
    for e_index in candidates[4]:
        valid_f = candidates[5][candidates[5] >= e_index]
        if not len(valid_f):
            continue
        f_index = int(valid_f[np.argmax(scores[valid_f, 5])])
        for g_index in candidates[6]:
            foundation_limit = min(int(e_index), int(g_index))
            chosen = []
            feasible = True
            for event_index in range(4):
                valid = candidates[event_index][
                    candidates[event_index] <= foundation_limit
                ]
                if not len(valid):
                    feasible = False
                    break
                chosen.append(int(valid[np.argmax(scores[valid, event_index])]))
            if not feasible:
                continue
            indices = chosen + [int(e_index), f_index, int(g_index)]
            total = sum(scores[index, event] for event, index in enumerate(indices))
            if total > best_score:
                best_score = float(total)
                best_indices = indices

    if best_indices is None:
        best_indices = [
            int(np.argmax(scores[:, event_index]))
            for event_index in range(len(EVENTS))
        ]
    return {
        event_name: int(frame_numbers[index])
        for (event_name, _), index in zip(EVENTS, best_indices)
    }


def expand_event_predictions(event_frames: dict[str, int]) -> dict[str, int]:
    expanded = {}
    for event_name, step_ids in EVENTS:
        frame = event_frames[event_name]
        if step_ids == (9, 12):
            expanded["9_or_12"] = frame
            continue
        for step_id in step_ids:
            expanded[str(step_id)] = frame
    return expanded


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
