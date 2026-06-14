from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from xgboost import XGBClassifier

from psr_geometry import (
    EVENT_NAMES,
    EVENTS,
    ROOT,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    decode_completion_events,
    event_completion_frames,
    expand_event_predictions,
    make_event_targets,
    make_state_targets,
    save_json,
    state_to_event_scores,
    temporal_tabular_features,
)


DEFAULT_FEATURES = ROOT / "data" / "geometric_features"
DEFAULT_OUTPUT = ROOT / "models" / "trained_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost and TCN PSR event models from YOLO geometry."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=int, default=15)
    parser.add_argument("--xgb-estimators", type=int, default=220)
    parser.add_argument("--tcn-epochs-cv", type=int, default=60)
    parser.add_argument("--tcn-epochs-final", type=int, default=80)
    parser.add_argument(
        "--target-mode",
        choices=("event", "state"),
        default="state",
        help="Train direct completion windows or cumulative completion states.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sequence(features_dir: Path, sequence: str, tolerance: int) -> dict:
    data = np.load(features_dir / f"{sequence}.npz")
    features = data["features"].astype(np.float32)
    frame_numbers = data["frame_numbers"].astype(np.int32)
    targets, present = make_event_targets(frame_numbers, sequence, tolerance)
    state_targets, state_valid, _ = make_state_targets(
        frame_numbers, sequence, tolerance
    )
    return {
        "sequence": sequence,
        "features": features,
        "tabular": temporal_tabular_features(features),
        "frames": frame_numbers,
        "targets": targets,
        "state_targets": state_targets,
        "state_valid": state_valid,
        "present": present,
    }


def xgb_parameters(args: argparse.Namespace, positive_count: int, total: int) -> dict:
    negative_count = max(1, total - positive_count)
    return {
        "n_estimators": args.xgb_estimators,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.65,
        "min_child_weight": 3,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_bin": 128,
        "device": "cuda",
        "scale_pos_weight": min(100.0, negative_count / max(1, positive_count)),
        "random_state": args.seed,
        "n_jobs": 4,
    }


def train_xgb(
    training: list[dict],
    args: argparse.Namespace,
) -> list[XGBClassifier]:
    x = np.concatenate([item["tabular"] for item in training])
    target_key = "targets" if args.target_mode == "event" else "state_targets"
    y = np.concatenate([item[target_key] for item in training])
    valid = np.concatenate(
        [
            np.ones_like(item["targets"])
            if args.target_mode == "event"
            else item["state_valid"]
            for item in training
        ]
    )
    models = []
    for event_index, event_name in enumerate(EVENT_NAMES):
        selected = valid[:, event_index] > 0
        event_x = x[selected]
        event_y = y[selected, event_index]
        positives = int(event_y.sum())
        model = XGBClassifier(
            **xgb_parameters(args, positives, len(event_y))
        )
        try:
            model.fit(event_x, event_y)
        except Exception as error:
            print(f"{event_name}: CUDA XGBoost failed ({error}); using CPU", flush=True)
            params = xgb_parameters(args, positives, len(event_y))
            params["device"] = "cpu"
            model = XGBClassifier(**params)
            model.fit(event_x, event_y)
        models.append(model)
        print(f"XGBoost trained: {event_name}", flush=True)
    return models


def predict_xgb(models: list[XGBClassifier], item: dict) -> np.ndarray:
    return np.column_stack(
        [model.predict_proba(item["tabular"])[:, 1] for model in models]
    ).astype(np.float32)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class GeometryTCN(nn.Module):
    def __init__(self, input_size: int, event_count: int):
        super().__init__()
        channels = 64
        self.network = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=1),
            nn.ReLU(),
            ResidualTCNBlock(channels, 1, 0.15),
            ResidualTCNBlock(channels, 2, 0.15),
            ResidualTCNBlock(channels, 4, 0.15),
            ResidualTCNBlock(channels, 8, 0.15),
            nn.Conv1d(channels, event_count, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def normalization(training: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([item["features"] for item in training])
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def train_tcn(
    training: list[dict],
    epochs: int,
    seed: int,
    target_mode: str,
) -> tuple[GeometryTCN, np.ndarray, np.ndarray]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalization(training)
    model = GeometryTCN(training[0]["features"].shape[1], len(EVENTS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    target_key = "targets" if target_mode == "event" else "state_targets"
    all_targets = np.concatenate([item[target_key] for item in training])
    all_valid = np.concatenate(
        [
            np.ones_like(item["targets"])
            if target_mode == "event"
            else item["state_valid"]
            for item in training
        ]
    )
    positives = (all_targets * all_valid).sum(axis=0)
    negatives = ((1.0 - all_targets) * all_valid).sum(axis=0)
    pos_weight = torch.as_tensor(
        np.clip(negatives / np.maximum(positives, 1), 1, 100),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight[:, None], reduction="none"
    )

    model.train()
    for epoch in range(epochs):
        order = np.random.permutation(len(training))
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for sequence_index in order:
            item = training[int(sequence_index)]
            x = (item["features"] - mean) / std
            inputs = torch.from_numpy(x.T[None]).to(device)
            targets = torch.from_numpy(item[target_key].T[None]).to(device)
            valid = (
                np.ones_like(item["targets"])
                if target_mode == "event"
                else item["state_valid"]
            )
            valid_tensor = torch.from_numpy(valid.T[None]).to(device)
            element_loss = criterion(model(inputs), targets)
            loss = (
                (element_loss * valid_tensor).sum()
                / valid_tensor.sum().clamp_min(1.0)
                / len(training)
            )
            loss.backward()
            epoch_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"TCN epoch {epoch + 1}/{epochs}: loss={epoch_loss:.5f}", flush=True)
    return model, mean, std


@torch.inference_mode()
def predict_tcn(
    model: GeometryTCN,
    mean: np.ndarray,
    std: np.ndarray,
    item: dict,
) -> np.ndarray:
    device = next(model.parameters()).device
    x = (item["features"] - mean) / std
    inputs = torch.from_numpy(x.T[None]).to(device)
    return torch.sigmoid(model(inputs))[0].T.cpu().numpy().astype(np.float32)


def evaluate_prediction(
    model_name: str,
    item: dict,
    probabilities: np.ndarray,
    tolerance: int,
    fold: str,
    target_mode: str,
) -> tuple[list[dict], dict]:
    event_probabilities = (
        probabilities
        if target_mode == "event"
        else state_to_event_scores(probabilities)
    )
    decoded = decode_completion_events(event_probabilities, item["frames"])
    truth = event_completion_frames(item["sequence"])
    rows = []
    aps = []
    hits = []
    errors = []
    for event_index, event_name in enumerate(EVENT_NAMES):
        if event_name not in truth:
            continue
        predicted = decoded[event_name]
        error = predicted - truth[event_name]
        hit = abs(error) <= tolerance
        ap = float(
            average_precision_score(
                item["targets"][:, event_index], probabilities[:, event_index]
                if target_mode == "event"
                else event_probabilities[:, event_index]
            )
        )
        aps.append(ap)
        hits.append(hit)
        errors.append(abs(error))
        rows.append(
            {
                "fold": fold,
                "model": model_name,
                "sequence": item["sequence"],
                "event": event_name,
                "true_frame": truth[event_name],
                "predicted_frame": predicted,
                "signed_error": error,
                "within_tolerance": int(hit),
                "frame_average_precision": round(ap, 6),
            }
        )
    summary = {
        "fold": fold,
        "model": model_name,
        "sequence": item["sequence"],
        "events_evaluated": len(rows),
        "tolerance_hit_rate": float(np.mean(hits)),
        "mean_absolute_frame_error": float(np.mean(errors)),
        "mean_frame_average_precision": float(np.mean(aps)),
    }
    return rows, summary


def save_probabilities(
    path: Path,
    item: dict,
    probabilities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_numbers=item["frames"],
        probabilities=probabilities,
        event_names=np.asarray(EVENT_NAMES),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_data = {
        sequence: load_sequence(args.features, sequence, args.tolerance)
        for sequence in TRAIN_SEQUENCES
    }
    test_data = {
        sequence: load_sequence(args.features, sequence, args.tolerance)
        for sequence in TEST_SEQUENCES
    }

    event_rows = []
    fold_summaries = []
    for fold_index, held_out in enumerate(TRAIN_SEQUENCES):
        print(f"\nLOSO fold: {held_out}", flush=True)
        training = [
            train_data[sequence]
            for sequence in TRAIN_SEQUENCES
            if sequence != held_out
        ]
        validation = train_data[held_out]

        xgb_models = train_xgb(training, args)
        xgb_probabilities = predict_xgb(xgb_models, validation)
        rows, summary = evaluate_prediction(
            "xgboost",
            validation,
            xgb_probabilities,
            args.tolerance,
            held_out,
            args.target_mode,
        )
        event_rows.extend(rows)
        fold_summaries.append(summary)
        save_probabilities(
            output / "cv_probabilities" / f"xgboost_{held_out}.npz",
            validation,
            xgb_probabilities,
        )

        tcn, mean, std = train_tcn(
            training,
            args.tcn_epochs_cv,
            args.seed + fold_index,
            args.target_mode,
        )
        tcn_probabilities = predict_tcn(tcn, mean, std, validation)
        rows, summary = evaluate_prediction(
            "tcn",
            validation,
            tcn_probabilities,
            args.tolerance,
            held_out,
            args.target_mode,
        )
        event_rows.extend(rows)
        fold_summaries.append(summary)
        save_probabilities(
            output / "cv_probabilities" / f"tcn_{held_out}.npz",
            validation,
            tcn_probabilities,
        )

    with (output / "loso_event_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    save_json(output / "loso_fold_summary.json", fold_summaries)

    aggregate = {}
    for model_name in ("xgboost", "tcn"):
        selected = [row for row in fold_summaries if row["model"] == model_name]
        aggregate[model_name] = {
            "tolerance_hit_rate": float(
                np.mean([row["tolerance_hit_rate"] for row in selected])
            ),
            "mean_absolute_frame_error": float(
                np.mean([row["mean_absolute_frame_error"] for row in selected])
            ),
            "mean_frame_average_precision": float(
                np.mean([row["mean_frame_average_precision"] for row in selected])
            ),
        }
    save_json(output / "loso_aggregate.json", aggregate)

    print("\nTraining final models on all four sequences", flush=True)
    all_training = [train_data[sequence] for sequence in TRAIN_SEQUENCES]
    final_xgb = train_xgb(all_training, args)
    xgb_dir = output / "xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    for event_name, model in zip(EVENT_NAMES, final_xgb):
        joblib.dump(model, xgb_dir / f"{event_name}.joblib")

    final_tcn, mean, std = train_tcn(
        all_training,
        args.tcn_epochs_final,
        args.seed + 100,
        args.target_mode,
    )
    tcn_dir = output / "tcn"
    tcn_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_tcn.state_dict(),
            "input_size": all_training[0]["features"].shape[1],
            "event_names": EVENT_NAMES,
            "mean": mean,
            "std": std,
            "target_mode": args.target_mode,
        },
        tcn_dir / "model.pt",
    )

    test_rows = []
    test_summaries = []
    for sequence, item in test_data.items():
        for model_name, probabilities in (
            ("xgboost", predict_xgb(final_xgb, item)),
            ("tcn", predict_tcn(final_tcn, mean, std, item)),
        ):
            save_probabilities(
                output / "test_probabilities" / f"{model_name}_{sequence}.npz",
                item,
                probabilities,
            )
            rows, summary = evaluate_prediction(
                model_name,
                item,
                probabilities,
                args.tolerance,
                "final_test",
                args.target_mode,
            )
            test_rows.extend(rows)
            test_summaries.append(summary)
            event_scores = (
                probabilities
                if args.target_mode == "event"
                else state_to_event_scores(probabilities)
            )
            event_frames = decode_completion_events(event_scores, item["frames"])
            save_json(
                output / "test_predictions" / f"{model_name}_{sequence}.json",
                {
                    "sequence": sequence,
                    "model": model_name,
                    "target_mode": args.target_mode,
                    "event_frames": event_frames,
                    "psr_label_code_frames": expand_event_predictions(event_frames),
                },
            )
    with (output / "test_event_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    save_json(output / "test_summary.json", test_summaries)
    print(json.dumps({"loso": aggregate, "test": test_summaries}, indent=2))


if __name__ == "__main__":
    main()
