"""Train and evaluate an XGBoost baseline on the fixed ELAsTiCC split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_PATH, TRAIN_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on train.parquet.")
    parser.add_argument("--train", default=TRAIN_PATH, help="Train parquet path.")
    parser.add_argument("--test", default=TEST_PATH, help="Test parquet path.")
    parser.add_argument(
        "--model-output",
        default=Path(TRAIN_PATH).parent / "xgboost_model.json",
        help="Where to save the trained XGBoost model.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads for XGBoost. Defaults to all CPU cores.",
    )
    return parser.parse_args()


def load_xy(path: str, label_encoder: LabelEncoder | None = None):
    df = pd.read_parquet(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    x = df[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y_labels = df[TARGET_COLUMN].astype(str).to_numpy()
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
    else:
        y = label_encoder.transform(y_labels)
    return x, y, label_encoder


def train_xgboost(train_path: str, test_path: str, model_output: str, n_jobs: int) -> XGBClassifier:
    x_train, y_train, label_encoder = load_xy(train_path)
    x_test, y_test, _ = load_xy(test_path, label_encoder)
    num_classes = len(label_encoder.classes_)

    model_params = {
        "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": max(1, n_jobs),
    }
    if num_classes > 2:
        model_params["num_class"] = num_classes

    model = XGBClassifier(**model_params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\nFeature Importance:")
    for feature_name, importance in importances:
        print(f"{feature_name}: {importance:.6f}")

    model.save_model(model_output)
    print(f"Saved model to {model_output}")
    return model


def main() -> int:
    args = parse_args()
    train_xgboost(str(args.train), str(args.test), str(args.model_output), args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
