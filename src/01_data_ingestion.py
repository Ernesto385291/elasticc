"""Join ELAsTiCC HEAD/PHOTOMETRY parquet files and keep g/r-band rows."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from config import CLASS_FILE_ALIASES, GBAND_RBAND_PHOTOMETRY_PATH, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single g/r-band ELAsTiCC photometry parquet file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing <class>_head.parquet and <class>_photometry.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GBAND_RBAND_PHOTOMETRY_PATH,
        help="Output parquet path for joined g/r-band photometry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="DuckDB worker threads to use. Defaults to all CPU cores.",
    )
    return parser.parse_args()


def class_slug_from_path(path: Path, suffix: str) -> str:
    return re.sub(f"{re.escape(suffix)}$", "", path.stem)


def class_name_from_slug(slug: str) -> str:
    normalized = slug.lower().replace("_", "-")
    return CLASS_FILE_ALIASES.get(normalized, slug.replace("-", " ").title())


def discover_class_pairs(input_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for head_path in sorted(input_dir.glob("*_head.parquet")):
        slug = class_slug_from_path(head_path, "_head")
        phot_path = input_dir / f"{slug}_photometry.parquet"
        if phot_path.exists():
            pairs.append((class_name_from_slug(slug), head_path, phot_path))
    return pairs


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_class_query(class_name: str, head_path: Path, phot_path: Path) -> str:
    return f"""
        SELECT
            p.SNID,
            h.SNTYPE,
            h.SIM_TYPE_INDEX,
            h.SIM_TYPE_NAME,
            {sql_literal(class_name)} AS target,
            p.MJD,
            p.BAND,
            p.FLUXCAL,
            p.FLUXCALERR,
            p.PHOTFLAG,
            p.PHOTPROB
        FROM read_parquet({sql_literal(phot_path)}) AS p
        INNER JOIN read_parquet({sql_literal(head_path)}) AS h
            ON CAST(p.SNID AS VARCHAR) = CAST(h.SNID AS VARCHAR)
        WHERE p.BAND IN ('g', 'r')
          AND p.FLUXCAL IS NOT NULL
          AND p.MJD IS NOT NULL
    """


def ingest_rband(
    input_dir: Path,
    output_path: Path,
    overwrite: bool = False,
    threads: int | None = None,
) -> Path:
    pairs = discover_class_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(f"No HEAD/PHOTOMETRY parquet pairs found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    if output_path.exists():
        output_path.unlink()

    con = duckdb.connect()
    try:
        if threads:
            con.execute(f"PRAGMA threads={max(1, threads)}")
        class_queries = []
        for class_name, head_path, phot_path in pairs:
            print(f"Processing {class_name}: {phot_path.name} + {head_path.name}")
            class_queries.append(build_class_query(class_name, head_path, phot_path))

        union_query = "\nUNION ALL\n".join(class_queries)
        con.execute(
            f"""
            COPY ({union_query})
            TO {sql_literal(output_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    schema = pq.read_schema(output_path)
    print(f"Saved {output_path} with columns: {', '.join(schema.names)}")
    return output_path


def main() -> int:
    args = parse_args()
    ingest_rband(args.input_dir, args.output, args.overwrite, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
