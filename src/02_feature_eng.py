"""Extract per-band statistical and Lomb-Scargle features from g/r light curves."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import kurtosis, skew

from config import (
    BASE_FEATURE_COLUMNS,
    FEATURES_PATH,
    GBAND_RBAND_PHOTOMETRY_PATH,
    SNID_COLUMN,
    TARGET_COLUMN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-SNID g/r-band features.")
    parser.add_argument("--input", default=GBAND_RBAND_PHOTOMETRY_PATH, help="Joined g/r-band parquet.")
    parser.add_argument("--output", default=FEATURES_PATH, help="Output feature parquet.")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum observations per SNID.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker processes for per-light-curve feature extraction.",
    )
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


def transient_features(mjd: np.ndarray, flux: np.ndarray) -> dict[str, float]:
    """Return baseline-relative transient metrics, using NaN when a phase is missed."""
    features = {
        "Rise_Time": np.nan,
        "Decay_Time": np.nan,
        "Peak_to_Median_Ratio": np.nan,
        "Max_Flux_Diff": np.nan,
    }

    try:
        if len(flux) < 2:
            return features

        peak_index = int(np.nanargmax(flux))
        peak_flux = float(flux[peak_index])
        peak_mjd = float(mjd[peak_index])
        median_flux = float(np.nanmedian(flux))
        baseline_flux = float(np.nanpercentile(flux, 10))
        peak_excess = peak_flux - baseline_flux

        if np.isfinite(median_flux) and median_flux != 0:
            features["Peak_to_Median_Ratio"] = peak_flux / median_flux

        flux_diff = np.abs(np.diff(flux))
        if len(flux_diff) > 0:
            features["Max_Flux_Diff"] = float(np.nanmax(flux_diff))

        pre_peak_flux = flux[: peak_index + 1]
        pre_peak_mjd = mjd[: peak_index + 1]
        if len(pre_peak_flux) >= 2 and np.isfinite(peak_excess) and peak_excess > 0:
            rise_threshold = baseline_flux + 0.1 * peak_excess
            rise_candidates = np.flatnonzero(pre_peak_flux >= rise_threshold)
            if len(rise_candidates) > 0:
                rise_mjd = float(pre_peak_mjd[int(rise_candidates[0])])
                if rise_mjd <= peak_mjd:
                    features["Rise_Time"] = peak_mjd - rise_mjd

        post_peak_flux = flux[peak_index + 1 :]
        post_peak_mjd = mjd[peak_index + 1 :]
        if len(post_peak_flux) > 0 and np.isfinite(peak_excess) and peak_excess > 0:
            decay_threshold = baseline_flux + 0.5 * peak_excess
            decay_candidates = np.flatnonzero(post_peak_flux <= decay_threshold)
            if len(decay_candidates) > 0:
                decay_mjd = float(post_peak_mjd[int(decay_candidates[0])])
                features["Decay_Time"] = decay_mjd - peak_mjd
    except Exception:
        return features

    return features


def band_features(mjd: np.ndarray, flux: np.ndarray, min_points: int) -> dict[str, float] | None:
    if len(flux) < min_points:
        return None

    order = np.argsort(mjd)
    mjd = mjd[order].astype(float, copy=False)
    flux = flux[order].astype(float, copy=False)
    valid = np.isfinite(mjd) & np.isfinite(flux)
    mjd = mjd[valid]
    flux = flux[valid]
    if len(flux) < min_points:
        return None

    mean_flux = float(np.mean(flux))
    std_flux = float(np.std(flux, ddof=1)) if len(flux) > 1 else 0.0
    features = {
        "Mean": mean_flux,
        "Std": std_flux,
        "Skew": float(skew(flux, bias=False, nan_policy="omit")),
        "Kurtosis": float(kurtosis(flux, bias=False, nan_policy="omit")),
        "Mean_Variance": float(std_flux / mean_flux) if mean_flux != 0 else np.nan,
        "Period": estimate_period(mjd, flux),
        "Amplitude": float((np.nanmax(flux) - np.nanmin(flux)) / 2.0),
    }
    features.update(transient_features(mjd, flux))
    return features


def extract_features_from_payload(
    payload: tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
) -> dict[str, object] | None:
    snid, target, mjd_g, flux_g, mjd_r, flux_r, min_points = payload
    features_g = band_features(mjd_g, flux_g, min_points)
    features_r = band_features(mjd_r, flux_r, min_points)
    if features_g is None or features_r is None:
        return None

    row: dict[str, object] = {
        SNID_COLUMN: snid,
        TARGET_COLUMN: target,
    }
    row.update({f"{feature}_g": features_g[feature] for feature in BASE_FEATURE_COLUMNS})
    row.update({f"{feature}_r": features_r[feature] for feature in BASE_FEATURE_COLUMNS})
    row["Color_g_r"] = features_g["Mean"] - features_r["Mean"]
    return row


def group_payloads(
    photometry: pd.DataFrame,
    min_points: int,
) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    payloads = []
    grouped = photometry.groupby([SNID_COLUMN, TARGET_COLUMN], sort=False)
    for (snid, target), group in grouped:
        g_band = group.loc[group["BAND"] == "g", ["MJD", "FLUXCAL"]]
        r_band = group.loc[group["BAND"] == "r", ["MJD", "FLUXCAL"]]
        if len(g_band) < min_points or len(r_band) < min_points:
            continue

        payloads.append(
            (
                str(snid),
                str(target),
                g_band["MJD"].to_numpy(dtype=float, copy=True),
                g_band["FLUXCAL"].to_numpy(dtype=float, copy=True),
                r_band["MJD"].to_numpy(dtype=float, copy=True),
                r_band["FLUXCAL"].to_numpy(dtype=float, copy=True),
                min_points,
            )
        )
    return payloads


def extract_features(
    input_path: str,
    output_path: str,
    min_points: int,
    workers: int | None = None,
) -> pd.DataFrame:
    photometry = pd.read_parquet(input_path)
    required_columns = {SNID_COLUMN, TARGET_COLUMN, "MJD", "BAND", "FLUXCAL"}
    missing = required_columns - set(photometry.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    photometry = photometry.loc[photometry["BAND"].isin(["g", "r"])].copy()
    payloads = group_payloads(photometry, min_points)
    worker_count = max(1, workers or 1)
    print(f"Extracting features for {len(payloads):,} light curves using {worker_count} workers")

    rows = []
    if worker_count == 1:
        for index, payload in enumerate(payloads, start=1):
            features = extract_features_from_payload(payload)
            if features is not None:
                rows.append(features)
            if index % 1000 == 0:
                print(f"Processed {index:,} light curves")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            chunk_size = max(1, len(payloads) // (worker_count * 16))
            results = executor.map(extract_features_from_payload, payloads, chunksize=chunk_size)
            for index, features in enumerate(results, start=1):
                if features is not None:
                    rows.append(features)
                if index % 1000 == 0:
                    print(f"Processed {index:,} light curves")

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df.to_parquet(output_path, index=False)
    print(f"Saved {len(feature_df):,} feature rows to {output_path}")
    return feature_df


def main() -> int:
    args = parse_args()
    extract_features(str(args.input), str(args.output), args.min_points, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
