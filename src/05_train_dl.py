"""Train a PyTorch MLP with validation early stopping."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_PATH, TRAIN_PATH, VAL_PATH


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch MLP on fixed ELAsTiCC split.")
    parser.add_argument("--train", default=TRAIN_PATH, help="Train parquet path.")
    parser.add_argument("--val", default=VAL_PATH, help="Validation parquet path.")
    parser.add_argument("--test", default=TEST_PATH, help="Test parquet path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="PyTorch CPU compute threads. Defaults to all CPU cores.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Tensor data is already in memory, so 0 is often fastest.",
    )
    parser.add_argument(
        "--model-output",
        default=Path(TRAIN_PATH).parent / "mlp_model.pt",
        help="Where to save the best validation checkpoint.",
    )
    return parser.parse_args()


def frame_to_xy(df: pd.DataFrame, label_encoder: LabelEncoder | None = None):
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_labels = df[TARGET_COLUMN].astype(str).to_numpy()
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
    else:
        y = label_encoder.transform(y_labels)
    return x, y.astype(np.int64), label_encoder


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    loader_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, loader_workers),
        persistent_workers=loader_workers > 0,
    )


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * len(x_batch)
            total_rows += len(x_batch)
    return total_loss / max(total_rows, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions)


def train_mlp(args: argparse.Namespace) -> MLPClassifier:
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(max(1, min(args.threads, 4)))

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)

    x_train, y_train, label_encoder = frame_to_xy(train_df)
    x_val, y_val, _ = frame_to_xy(val_df, label_encoder)
    x_test, y_test, _ = frame_to_xy(test_df, label_encoder)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=x_train.shape[1], num_classes=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True, loader_workers=args.loader_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False, loader_workers=args.loader_workers)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, loader_workers=args.loader_workers)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(x_batch)
            total_rows += len(x_batch)

        train_loss = total_loss / max(total_rows, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "feature_columns": FEATURE_COLUMNS,
                    "classes": label_encoder.classes_.tolist(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "val_loss": best_val_loss,
                },
                args.model_output,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_predictions = predict(model, test_loader, device)
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )
    print(f"Saved best checkpoint to {args.model_output}")
    return model


def main() -> int:
    args = parse_args()
    train_mlp(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
