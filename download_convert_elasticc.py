#!/usr/bin/env python3
"""
Download ELAsTiCC training FITS shards and convert them into parquet datasets.

Recommended output layout:
  processed/
    cepheid_head.parquet
    cepheid_photometry.parquet
    rrl_head.parquet
    rrl_photometry.parquet
    eb_head.parquet
    eb_photometry.parquet
    d-sct_head.parquet
    d-sct_photometry.parquet

This is cleaner than keeping thousands of per-object files locally. The script
can optionally keep the raw FITS shards, but by default it streams each shard
through a temporary file and only keeps the parquet outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table


BASE_URL = (
    "https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/"
    "ELASTICC2_TRAINING_SAMPLE_2/"
)

DEFAULT_CLASS_ALIASES = {
    "cepheid": "ELASTICC2_TRAIN_02_Cepheid",
    "rrl": "ELASTICC2_TRAIN_02_RRL",
    "rrlyrae": "ELASTICC2_TRAIN_02_RRL",
    "eb": "ELASTICC2_TRAIN_02_EB",
    "d-sct": "ELASTICC2_TRAIN_02_d-Sct",
    "dsct": "ELASTICC2_TRAIN_02_d-Sct",
    "delta-scuti": "ELASTICC2_TRAIN_02_d-Sct",
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class ClassSpec:
    remote_dir: str
    output_slug: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ELAsTiCC class shards and convert them to parquet."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["cepheid", "rrl", "eb", "d-sct"],
        help=(
            "Classes to download. Accepts aliases like 'cepheid', 'rrl', 'eb', "
            "'d-sct', remote directory names like 'ELASTICC2_TRAIN_02_Cepheid', "
            "or 'all' to discover every class directory."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Base ELAsTiCC training sample URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Directory where parquet outputs will be written.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw"),
        help="Directory where raw FITS shards will be kept if --keep-raw is set.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded FITS.gz files instead of using temporary files only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet outputs.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Optional limit for testing on only the first N shard pairs per class.",
    )
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def parse_links(url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(fetch_text(url))
    return parser.links


def discover_remote_class_dirs(base_url: str) -> list[str]:
    dirs = []
    for href in parse_links(base_url):
        if href.endswith("/") and href.startswith("ELASTICC2_TRAIN_02_"):
            dirs.append(href.rstrip("/"))
    return sorted(set(dirs))


def normalize_slug(name: str) -> str:
    suffix = name.replace("ELASTICC2_TRAIN_02_", "")
    suffix = suffix.lower()
    suffix = re.sub(r"[^a-z0-9]+", "-", suffix).strip("-")
    return suffix


def resolve_classes(requested: Iterable[str], base_url: str) -> list[ClassSpec]:
    requested = list(requested)
    if any(item.lower() == "all" for item in requested):
        return [
            ClassSpec(remote_dir=remote_dir, output_slug=normalize_slug(remote_dir))
            for remote_dir in discover_remote_class_dirs(base_url)
        ]

    resolved: list[ClassSpec] = []
    for item in requested:
        key = item.strip()
        low = key.lower()
        remote_dir = DEFAULT_CLASS_ALIASES.get(low, key)
        output_slug = (
            low
            if low in {"cepheid", "rrl", "eb", "d-sct"}
            else normalize_slug(remote_dir)
        )
        resolved.append(ClassSpec(remote_dir=remote_dir, output_slug=output_slug))
    return resolved


def shard_pairs_for_class(class_url: str) -> list[tuple[str, str]]:
    head_files = {}
    phot_files = {}

    for href in parse_links(class_url):
        filename = href.rsplit("/", 1)[-1]
        if filename.endswith("_HEAD.FITS.gz"):
            prefix = filename[: -len("_HEAD.FITS.gz")]
            head_files[prefix] = filename
        elif filename.endswith("_PHOT.FITS.gz"):
            prefix = filename[: -len("_PHOT.FITS.gz")]
            phot_files[prefix] = filename

    missing = sorted(set(head_files) ^ set(phot_files))
    if missing:
        raise RuntimeError(f"Unpaired shards found in {class_url}: {missing[:5]}")

    return [
        (head_files[prefix], phot_files[prefix])
        for prefix in sorted(head_files)
    ]


def ensure_output_paths(
    output_dir: Path, output_slug: str, overwrite: bool
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / f"{output_slug}_head.parquet"
    phot_path = output_dir / f"{output_slug}_photometry.parquet"

    if not overwrite:
        existing = [path for path in [head_path, phot_path] if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {joined}. Use --overwrite to replace them."
            )

    for path in [head_path, phot_path]:
        if path.exists():
            path.unlink()

    return head_path, phot_path


def download_to_path(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    with urllib.request.urlopen(url, timeout=600) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return path


def localize_file(url: str, target_dir: Path | None) -> tuple[Path, bool]:
    filename = Path(urllib.parse.urlparse(url).path).name
    if target_dir is not None:
        return download_to_path(url, target_dir / filename), False

    handle = tempfile.NamedTemporaryFile(suffix=filename, delete=False)
    temp_path = Path(handle.name)
    with handle:
        with urllib.request.urlopen(url, timeout=600) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return temp_path, True


def fits_to_dataframe(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=False) as hdul:
        df = Table(hdul[1].data).to_pandas()
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]):
                df[column] = df[column].astype(str).str.strip()
        return df


def build_photometry_with_snid(head_df: pd.DataFrame, phot_df: pd.DataFrame) -> pd.DataFrame:
    phot_len = len(phot_df)
    snid_values = np.empty(phot_len, dtype=object)
    valid_mask = np.zeros(phot_len, dtype=bool)

    for row in head_df[["SNID", "PTROBS_MIN", "PTROBS_MAX"]].itertuples(index=False):
        start = int(row.PTROBS_MIN) - 1
        end = int(row.PTROBS_MAX)
        snid_values[start:end] = row.SNID
        valid_mask[start:end] = True

    phot_clean = phot_df.loc[valid_mask].copy()
    phot_clean.insert(0, "SNID", snid_values[valid_mask])
    return phot_clean.reset_index(drop=True)


def write_parquet_chunk(
    writer: pq.ParquetWriter | None,
    df: pd.DataFrame,
    path: Path,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def convert_class(
    class_spec: ClassSpec,
    base_url: str,
    output_dir: Path,
    raw_dir: Path,
    keep_raw: bool,
    overwrite: bool,
    limit_shards: int | None,
) -> tuple[Path, Path]:
    class_url = urllib.parse.urljoin(base_url, class_spec.remote_dir + "/")
    print(f"\n=== Converting {class_spec.remote_dir} ===")
    print(f"Source: {class_url}")

    shard_pairs = shard_pairs_for_class(class_url)
    if limit_shards is not None:
        shard_pairs = shard_pairs[:limit_shards]
    print(f"Shard pairs found: {len(shard_pairs)}")

    head_out, phot_out = ensure_output_paths(output_dir, class_spec.output_slug, overwrite)
    head_writer: pq.ParquetWriter | None = None
    phot_writer: pq.ParquetWriter | None = None

    raw_target_dir = raw_dir / class_spec.output_slug if keep_raw else None
    total_head_rows = 0
    total_phot_rows = 0

    try:
        for index, (head_name, phot_name) in enumerate(shard_pairs, start=1):
            print(f"[{index}/{len(shard_pairs)}] {head_name} + {phot_name}")
            head_url = urllib.parse.urljoin(class_url, head_name)
            phot_url = urllib.parse.urljoin(class_url, phot_name)

            head_path, head_is_temp = localize_file(head_url, raw_target_dir)
            phot_path, phot_is_temp = localize_file(phot_url, raw_target_dir)

            try:
                head_df = fits_to_dataframe(head_path)
                phot_df = fits_to_dataframe(phot_path)
                phot_df = build_photometry_with_snid(head_df, phot_df)

                head_writer = write_parquet_chunk(head_writer, head_df, head_out)
                phot_writer = write_parquet_chunk(phot_writer, phot_df, phot_out)

                total_head_rows += len(head_df)
                total_phot_rows += len(phot_df)
                print(
                    f"    wrote {len(head_df):,} head rows and {len(phot_df):,} phot rows "
                    f"(running totals: {total_head_rows:,} / {total_phot_rows:,})"
                )
            finally:
                if head_is_temp and head_path.exists():
                    head_path.unlink()
                if phot_is_temp and phot_path.exists():
                    phot_path.unlink()
    finally:
        if head_writer is not None:
            head_writer.close()
        if phot_writer is not None:
            phot_writer.close()

    print(f"Saved: {head_out}")
    print(f"Saved: {phot_out}")
    return head_out, phot_out


def main() -> int:
    args = parse_args()
    class_specs = resolve_classes(args.classes, args.base_url)

    print("Classes to process:")
    for spec in class_specs:
        print(f"  - {spec.remote_dir} -> {spec.output_slug}")

    output_paths = []
    for spec in class_specs:
        output_paths.append(
            convert_class(
                class_spec=spec,
                base_url=args.base_url,
                output_dir=args.output_dir,
                raw_dir=args.raw_dir,
                keep_raw=args.keep_raw,
                overwrite=args.overwrite,
                limit_shards=args.limit_shards,
            )
        )

    print("\nDone.")
    for head_path, phot_path in output_paths:
        print(f"  - {head_path}")
        print(f"  - {phot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
