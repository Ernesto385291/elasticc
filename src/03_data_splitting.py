"""Create exact train/test/validation datasets from the required class counts."""

from __future__ import annotations

import argparse

import pandas as pd

from config import (
    DATA_SPLITS,
    FEATURES_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_ALIASES,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split features using fixed per-class counts.")
    parser.add_argument("--input", default=FEATURES_PATH, help="Feature parquet path.")
    parser.add_argument("--train-output", default=TRAIN_PATH, help="Train parquet output.")
    parser.add_argument("--test-output", default=TEST_PATH, help="Test parquet output.")
    parser.add_argument("--val-output", default=VAL_PATH, help="Validation parquet output.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Sampling random seed.")
    parser.add_argument(
        "--require-all-classes",
        action="store_true",
        help="Fail if any class from DATA_SPLITS is absent. By default, split only present classes.",
    )
    return parser.parse_args()


def sample_exact_splits(
    features: pd.DataFrame,
    seed: int,
    require_all_classes: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_classes = sorted(set(DATA_SPLITS) - set(features[TARGET_COLUMN].unique()))
    if missing_classes and require_all_classes:
        raise ValueError(f"Missing classes in feature file: {missing_classes}")

    train_parts = []
    test_parts = []
    val_parts = []

    for offset, (class_name, split_counts) in enumerate(DATA_SPLITS.items()):
        class_df = features.loc[features[TARGET_COLUMN] == class_name]
        if class_df.empty:
            print(f"Skipping {class_name}: no rows present yet")
            continue

        required = split_counts["train"] + split_counts["test"] + split_counts["val"]
        if len(class_df) < required:
            print(
                f"Using all available rows for {class_name}: {len(class_df):,} available, "
                f"{required:,} required by the final table."
            )
            class_counts = proportional_counts(len(class_df), split_counts)
        else:
            class_counts = split_counts

        train_sample = class_df.sample(
            n=class_counts["train"],
            replace=False,
            random_state=seed + offset,
        )
        remaining = class_df.drop(index=train_sample.index)
        test_sample = remaining.sample(
            n=class_counts["test"],
            replace=False,
            random_state=seed + 1000 + offset,
        )
        remaining = remaining.drop(index=test_sample.index)
        val_sample = remaining.sample(
            n=class_counts["val"],
            replace=False,
            random_state=seed + 2000 + offset,
        )

        train_parts.append(train_sample)
        test_parts.append(test_sample)
        val_parts.append(val_sample)

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=seed)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=seed)
    return train_df, test_df, val_df


def proportional_counts(available_rows: int, target_counts: dict[str, int]) -> dict[str, int]:
    """Scale the required table counts down when working with partial local data."""
    total_required = sum(target_counts.values())
    raw = {
        split: available_rows * count / total_required
        for split, count in target_counts.items()
    }
    counts = {split: int(value) for split, value in raw.items()}

    for split in ["train", "test", "val"]:
        if target_counts[split] > 0 and available_rows >= 3 and counts[split] == 0:
            counts[split] = 1

    while sum(counts.values()) > available_rows:
        split = max(counts, key=counts.get)
        counts[split] -= 1

    remainders = sorted(
        raw,
        key=lambda split: raw[split] - int(raw[split]),
        reverse=True,
    )
    index = 0
    while sum(counts.values()) < available_rows:
        counts[remainders[index % len(remainders)]] += 1
        index += 1

    return counts


def write_splits(
    input_path: str,
    train_output: str,
    test_output: str,
    val_output: str,
    seed: int,
    require_all_classes: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(input_path)
    if TARGET_COLUMN not in features.columns:
        raise ValueError(f"Feature file must contain '{TARGET_COLUMN}' column")
    features = features.copy()
    features[TARGET_COLUMN] = features[TARGET_COLUMN].replace(TARGET_ALIASES)
    present_classes = set(features[TARGET_COLUMN].unique())
    unknown_classes = sorted(present_classes - set(DATA_SPLITS))
    if unknown_classes:
        counts = features.loc[features[TARGET_COLUMN].isin(unknown_classes), TARGET_COLUMN].value_counts()
        print("Warning: labels not present in DATA_SPLITS and ignored by the sampler:")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count:,} rows")

    train_df, test_df, val_df = sample_exact_splits(features, seed, require_all_classes)
    train_df.to_parquet(train_output, index=False)
    test_df.to_parquet(test_output, index=False)
    val_df.to_parquet(val_output, index=False)

    print(f"Saved train: {len(train_df):,} rows -> {train_output}")
    print(f"Saved test:  {len(test_df):,} rows -> {test_output}")
    print(f"Saved val:   {len(val_df):,} rows -> {val_output}")
    return train_df, test_df, val_df


def main() -> int:
    args = parse_args()
    write_splits(
        args.input,
        args.train_output,
        args.test_output,
        args.val_output,
        args.seed,
        args.require_all_classes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
