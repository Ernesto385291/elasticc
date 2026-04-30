"""Central configuration for the ELAsTiCC variable-star classifier."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"

GBAND_RBAND_PHOTOMETRY_PATH = PROCESSED_DIR / "elasticc_grband_photometry.parquet"
RBAND_PHOTOMETRY_PATH = GBAND_RBAND_PHOTOMETRY_PATH
FEATURES_PATH = PROCESSED_DIR / "features_extracted.parquet"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"

RANDOM_STATE = 42
TARGET_COLUMN = "target"
SNID_COLUMN = "SNID"
BASE_FEATURE_COLUMNS = [
    "Mean",
    "Std",
    "Skew",
    "Kurtosis",
    "Mean_Variance",
    "Period",
    "Amplitude",
    "Rise_Time",
    "Decay_Time",
    "Peak_to_Median_Ratio",
    "Max_Flux_Diff",
]
FEATURE_COLUMNS = [
    *(f"{feature}_g" for feature in BASE_FEATURE_COLUMNS),
    *(f"{feature}_r" for feature in BASE_FEATURE_COLUMNS),
    "Color_g_r",
]

# Required fixed class distribution. Do not replace with percentage splits.
DATA_SPLITS = {
    "AGN": {"train": 38113, "test": 20000, "val": 1887},
    "CaRT": {"train": 7807, "test": 3517, "val": 400},
    "Cepheid": {"train": 13088, "test": 5901, "val": 683},
    "Delta Scuti": {"train": 19611, "test": 8849, "val": 1039},
    "Dwarf Novae": {"train": 7608, "test": 3439, "val": 417},
    "EB": {"train": 38036, "test": 20000, "val": 1964},
    "ILOT": {"train": 7090, "test": 3197, "val": 371},
    "KN": {"train": 4211, "test": 1896, "val": 215},
    "M-dwarf Flare": {"train": 1780, "test": 796, "val": 79},
    "PISN": {"train": 37996, "test": 20000, "val": 2004},
    "RR Lyrae": {"train": 13278, "test": 6014, "val": 755},
    "SLSN": {"train": 37993, "test": 20000, "val": 2007},
    "SNI91bg": {"train": 27207, "test": 12272, "val": 1430},
    "SNII": {"train": 38018, "test": 20000, "val": 1982},
    "SNIa": {"train": 38000, "test": 20000, "val": 2000},
    "SNIax": {"train": 26610, "test": 12012, "val": 1420},
    "SNIb/c": {"train": 37935, "test": 20000, "val": 2065},
    "TDE": {"train": 38023, "test": 20000, "val": 1977},
    "uLens": {"train": 16652, "test": 7537, "val": 940},
}

CLASS_FILE_ALIASES = {
    "agn": "AGN",
    "cart": "CaRT",
    "cepheid": "Cepheid",
    "d-sct": "Delta Scuti",
    "dsct": "Delta Scuti",
    "delta-scuti": "Delta Scuti",
    "dwarf-novae": "Dwarf Novae",
    "dwarf-nova": "Dwarf Novae",
    "eb": "EB",
    "ilot": "ILOT",
    "kn": "KN",
    "m-dwarf-flare": "M-dwarf Flare",
    "mdwarf-flare": "M-dwarf Flare",
    "pisn": "PISN",
    "rrl": "RR Lyrae",
    "rr-lyrae": "RR Lyrae",
    "rrlyrae": "RR Lyrae",
    "slsn": "SLSN",
    "sni91bg": "SNI91bg",
    "sni-91bg": "SNI91bg",
    "snia-91bg": "SNI91bg",
    "snia-91bg-salt3": "SNI91bg",
    "snii": "SNII",
    "sn-ii": "SNII",
    "snii-nmf": "SNII",
    "snia": "SNIa",
    "sn-ia": "SNIa",
    "snia-salt3": "SNIa",
    "sniax": "SNIax",
    "snia-x": "SNIax",
    "sniax-salt3": "SNIax",
    "snib-c": "SNIb/c",
    "snibc": "SNIb/c",
    "snibc-nmf": "SNIb/c",
    "sn-ibc": "SNIb/c",
    "tde": "TDE",
    "ulens": "uLens",
}

TARGET_ALIASES = {
    "Eclipsing Binary": "EB",
    "D-Sct": "Delta Scuti",
    "d-Sct": "Delta Scuti",
    "RRL": "RR Lyrae",
    "RRLYR": "RR Lyrae",
    "Snia": "SNIa",
    "Snia Salt3": "SNIa",
    "Sn Ia": "SNIa",
    "Sni91Bg": "SNI91bg",
    "Sni 91Bg": "SNI91bg",
    "Snia 91Bg": "SNI91bg",
    "Snia 91Bg Salt3": "SNI91bg",
    "Snii": "SNII",
    "Sn Ii": "SNII",
    "Snii Nmf": "SNII",
    "Sniax": "SNIax",
    "Sniax Salt3": "SNIax",
    "Snib C": "SNIb/c",
    "Snibc": "SNIb/c",
    "Snibc Nmf": "SNIb/c",
}
