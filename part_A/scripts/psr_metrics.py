from __future__ import annotations

from itertools import combinations

import numpy as np

from psr_geometry import EVENTS


FPS = 10.0
PROCEDURE_STEP_BY_PSR_CODE = {
    3: 1,
    6: 2,
    9: 3,
    12: 3,
    15: 4,
    18: 5,
    21: 6,
    24: 7,
    27: 8,
    30: 9,
}
EVENT_LABELS = {
    "front_chassis_pair": "Procedure steps 1+2: front chassis + pin",
    "rear_chassis_variant": "Procedure step 3: rear chassis",
    "front_rear_chassis_pin": "Procedure step 4: front-rear chassis pin",
    "rear_rear_chassis_pin": "Procedure step 5: rear-rear chassis pin",
    "front_bracket_pair": "Procedure steps 6+7: bracket + screw",
    "front_wheel_assy": "Procedure step 8: front wheel assembly",
    "rear_wheel_assy": "Procedure step 9: rear wheel assembly",
}
EVENT_PROCEDURE_STEP_COUNTS = {
    "front_chassis_pair": 2,
    "rear_chassis_variant": 1,
    "front_rear_chassis_pin": 1,
    "rear_rear_chassis_pin": 1,
    "front_bracket_pair": 2,
    "front_wheel_assy": 1,
    "rear_wheel_assy": 1,
}


def expand_recognitions(
    recognized: dict[str, int],
    true_step_ids: set[int],
) -> list[dict]:
    predictions = []
    for event_name, step_ids in EVENTS:
        if event_name not in recognized:
            continue
        for step_id in step_ids:
            if step_id in true_step_ids:
                predictions.append(
                    {
                        "frame": recognized[event_name],
                        "procedure_step": PROCEDURE_STEP_BY_PSR_CODE[step_id],
                        "psr_label_code": step_id,
                        "event": event_name,
                    }
                )
    return sorted(predictions, key=lambda row: (row["frame"], row["procedure_step"]))


def event_timing_metrics(
    ground_truth: dict[str, int],
    predictions: dict[str, int],
) -> dict:
    event_names = list(ground_truth)
    predicted_event_names = [name for name in event_names if name in predictions]
    missing = sorted(set(event_names) - set(predictions))

    per_event = []
    absolute_errors_seconds = []
    true_positives = 0
    false_positives = 0
    false_negatives = sum(EVENT_PROCEDURE_STEP_COUNTS[name] for name in missing)
    for event_name in predicted_event_names:
        true_frame = ground_truth[event_name]
        predicted_frame = predictions[event_name]
        error_frames = predicted_frame - true_frame
        error_seconds = error_frames / FPS
        represented_steps = EVENT_PROCEDURE_STEP_COUNTS[event_name]
        if error_frames >= 0:
            true_positives += represented_steps
            paper_status = "TP"
        else:
            false_positives += represented_steps
            paper_status = "FP"
        absolute_errors_seconds.append(abs(error_seconds))
        per_event.append(
            {
                "event": event_name,
                "label": EVENT_LABELS[event_name],
                "ground_truth_frame": true_frame,
                "predicted_frame": predicted_frame,
                "signed_error_frames": error_frames,
                "signed_error_seconds": error_seconds,
                "absolute_error_seconds": abs(error_seconds),
                "represented_procedure_steps": represented_steps,
                "paper_f1_status": paper_status,
            }
        )

    pairwise = []
    squared_interval_errors = []
    for first, second in combinations(predicted_event_names, 2):
        true_interval = abs(ground_truth[second] - ground_truth[first]) / FPS
        predicted_interval = abs(predictions[second] - predictions[first]) / FPS
        interval_error = predicted_interval - true_interval
        squared_error = interval_error**2
        squared_interval_errors.append(squared_error)
        pairwise.append(
            {
                "first_event": first,
                "second_event": second,
                "ground_truth_interval_seconds": true_interval,
                "predicted_interval_seconds": predicted_interval,
                "interval_error_seconds": interval_error,
                "squared_interval_error_seconds2": squared_error,
            }
        )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "event_count": len(predicted_event_names),
        "pair_count": len(pairwise),
        "completion_time_mae_seconds": (
            float(np.mean(absolute_errors_seconds))
            if absolute_errors_seconds
            else None
        ),
        "pairwise_completion_interval_rmse_seconds": float(
            np.sqrt(np.mean(squared_interval_errors))
        ) if squared_interval_errors else None,
        "paper_style_procedure_step_f1": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rule": "On-time or late = TP; premature = FP; missing completed step = FN.",
        },
        "per_event": per_event,
        "pairwise_intervals": pairwise,
    }
