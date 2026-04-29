"""Extract statistical and Lomb-Scargle features from r-band light curves."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import kurtosis, skew

from config import FEATURES_PATH, RBAND_PHOTOMETRY_PATH, SNID_COLUMN, TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-SNID r-band features.")
    parser.add_argument("--input", default=RBAND_PHOTOMETRY_PATH, help="Joined r-band parquet.")
    parser.add_argument("--output", default=FEATURES_PATH, help="Output feature parquet.")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum observations per SNID.")
    return parser.parse_args()


def estimate_period(mjd: np.ndarray, flux: np.ndarray) -> float:
    if len(mjd) < 5 or np.allclose(flux, flux[0]):
        return np.nan

    baseline = float(np.nanmax(mjd) - np.nanmin(mjd))
    if baseline <= 0:
        return np.nan

    min_frequency = 1.0 / baseline
    max_frequency = 10.0

    try:
        frequency, power = LombScargle(mjd, flux).autopower(
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            samples_per_peak=5,
        )
    except Exception:
        return np.nan

    if len(frequency) == 0:
        return np.nan

    best_frequency = float(frequency[np.nanargmax(power)])
    if best_frequency <= 0:
        return np.nan
    return 1.0 / best_frequency


def extract_features_for_group(snid: str, target: str, group: pd.DataFrame) -> dict[str, object] | None:
    clean = group[["MJD", "FLUXCAL"]].dropna().sort_values("MJD")
    if len(clean) < 5:
        return None

    flux = clean["FLUXCAL"].to_numpy(dtype=float)
    mjd = clean["MJD"].to_numpy(dtype=float)
    mean_flux = float(np.mean(flux))
    std_flux = float(np.std(flux, ddof=1)) if len(flux) > 1 else 0.0

    return {
        SNID_COLUMN: snid,
        TARGET_COLUMN: target,
        "Mean": mean_flux,
        "Std": std_flux,
        "Skew": float(skew(flux, bias=False, nan_policy="omit")),
        "Kurtosis": float(kurtosis(flux, bias=False, nan_policy="omit")),
        "Mean_Variance": float(std_flux / mean_flux) if mean_flux != 0 else np.nan,
        "Period": estimate_period(mjd, flux),
        "Amplitude": float((np.nanmax(flux) - np.nanmin(flux)) / 2.0),
    }


def extract_features(input_path: str, output_path: str, min_points: int) -> pd.DataFrame:
    photometry = pd.read_parquet(input_path)
    required_columns = {SNID_COLUMN, TARGET_COLUMN, "MJD", "FLUXCAL"}
    missing = required_columns - set(photometry.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    counts = photometry.groupby(SNID_COLUMN)["FLUXCAL"].transform("count")
    photometry = photometry.loc[counts >= min_points].copy()

    rows = []
    grouped = photometry.groupby([SNID_COLUMN, TARGET_COLUMN], sort=False)
    for index, ((snid, target), group) in enumerate(grouped, start=1):
        features = extract_features_for_group(snid, target, group)
        if features is not None:
            rows.append(features)
        if index % 1000 == 0:
            print(f"Processed {index:,} light curves")

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    feature_df.to_parquet(output_path, index=False)
    print(f"Saved {len(feature_df):,} feature rows to {output_path}")
    return feature_df


def main() -> int:
    args = parse_args()
    extract_features(str(args.input), str(args.output), args.min_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
