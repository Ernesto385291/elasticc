"""Train and evaluate an XGBoost baseline on the fixed ELAsTiCC split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_PATH, TRAIN_PATH, VAL_PATH


ABLATION_GROUPS = {
    "all": FEATURE_COLUMNS,
    "no_transient": [
        feature
        for feature in FEATURE_COLUMNS
        if not any(
            token in feature
            for token in ["Rise_Time", "Decay_Time", "Peak_to_Median_Ratio", "Max_Flux_Diff"]
        )
    ],
    "no_period": [feature for feature in FEATURE_COLUMNS if not feature.startswith("Period_")],
    "no_color": [feature for feature in FEATURE_COLUMNS if feature != "Color_g_r"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on train.parquet.")
    parser.add_argument("--train", default=TRAIN_PATH, help="Train parquet path.")
    parser.add_argument("--val", default=VAL_PATH, help="Validation parquet path for early stopping.")
    parser.add_argument("--test", default=TEST_PATH, help="Test parquet path.")
    parser.add_argument(
        "--features",
        choices=sorted(ABLATION_GROUPS),
        default="all",
        help="Feature set for ablation experiments.",
    )
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--model-output",
        default=Path(TRAIN_PATH).parent / "xgboost_model.json",
        help="Where to save the trained XGBoost model.",
    )
    parser.add_argument(
        "--metrics-output",
        default=Path(TRAIN_PATH).parent / "xgboost_metrics.json",
        help="Where to save metrics, confusion matrix, and feature importance.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads for XGBoost. Defaults to all CPU cores.",
    )
    return parser.parse_args()


def load_xy(path: str, feature_columns: list[str], label_encoder: LabelEncoder | None = None):
    df = pd.read_parquet(path)
    missing = set(feature_columns + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    x = df[feature_columns].to_numpy(dtype="float32")
    y_labels = df[TARGET_COLUMN].astype(str).to_numpy()
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
    else:
        y = label_encoder.transform(y_labels)
    return x, y, label_encoder


def train_xgboost(args: argparse.Namespace) -> XGBClassifier:
    feature_columns = ABLATION_GROUPS[args.features]
    x_train, y_train, label_encoder = load_xy(str(args.train), feature_columns)
    x_val, y_val, _ = load_xy(str(args.val), feature_columns, label_encoder)
    x_test, y_test, _ = load_xy(str(args.test), feature_columns, label_encoder)
    num_classes = len(label_encoder.classes_)

    model_params = {
        "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
        "n_estimators": args.n_estimators,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "reg_lambda": 2.0,
        "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": max(1, args.n_jobs),
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    if num_classes > 2:
        model_params["num_class"] = num_classes

    model = XGBClassifier(**model_params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    predictions = model.predict(x_test)
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions)
    matrix_df = pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("\nConfusion Matrix:")
    print(matrix_df.to_string())

    importances = sorted(
        zip(feature_columns, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\nFeature Importance:")
    for feature_name, importance in importances:
        print(f"{feature_name}: {importance:.6f}")

    metrics = {
        "feature_set": args.features,
        "feature_columns": feature_columns,
        "classes": label_encoder.classes_.tolist(),
        "accuracy": accuracy_score(y_test, predictions),
        "best_iteration": getattr(model, "best_iteration", None),
        "classification_report": report,
        "confusion_matrix": matrix_df.to_dict(),
        "feature_importance": [
            {"feature": feature_name, "importance": float(importance)}
            for feature_name, importance in importances
        ],
    }
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_model(args.model_output)
    print(f"Saved metrics to {args.metrics_output}")
    print(f"Saved model to {args.model_output}")
    return model


def main() -> int:
    args = parse_args()
    train_xgboost(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
