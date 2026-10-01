"""TSV loading and compact integer entity-id encoding.

Entity ids look like ``S1-925783039``.  Source-1 ids are stored as the bare
integer; Source-2/3 ids as ``source * 10**10 + number`` so a single int64 column
identifies a target record and its source.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

TARGET_BASE = 10_000_000_000

#: dataset root (contains train/ and test/); override with the ER_DATA_DIR env var
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", "student_resource/dataset"))


def read_source(path: str | Path) -> pl.DataFrame:
    """Read one ``*_sourceN.tsv`` file with every column as string (no quoting)."""
    return pl.read_csv(
        str(path),
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=False,
    )


def encode_s1(col: str = "entity_id") -> pl.Expr:
    return pl.col(col).str.slice(3).cast(pl.Int64)


def encode_target(col: str = "entity_id") -> pl.Expr:
    src = pl.col(col).str.slice(1, 1).cast(pl.Int64)
    return src * TARGET_BASE + pl.col(col).str.slice(3).cast(pl.Int64)


def decode_s1(col: str) -> pl.Expr:
    return pl.lit("S1-") + pl.col(col).cast(pl.Utf8)


def decode_target(col: str) -> pl.Expr:
    src = (pl.col(col) // TARGET_BASE).cast(pl.Utf8)
    return pl.lit("S") + src + pl.lit("-") + (pl.col(col) % TARGET_BASE).cast(pl.Utf8)


def load_s1(data_dir: str | Path, split: str) -> pl.DataFrame:
    df = read_source(Path(data_dir) / split / f"{split}_source1.tsv")
    return df.with_columns(encode_s1().alias("id"))


def load_targets(data_dir: str | Path, split: str) -> pl.DataFrame:
    parts = [
        read_source(Path(data_dir) / split / f"{split}_source{k}.tsv") for k in (2, 3)
    ]
    df = pl.concat(parts, how="vertical")
    return df.with_columns(encode_target().alias("id"))


def load_ground_truth(data_dir: str | Path) -> pl.DataFrame:
    """Return long-form ``(s1, tid)`` integer pairs from the training ground truth."""
    gt = read_source(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return (
        gt.select(
            encode_s1("source1_entity_id").alias("s1"),
            pl.col("matched_entity_ids").str.split(",").alias("tid"),
        )
        .explode("tid")
        .filter(pl.col("tid").is_not_null() & (pl.col("tid") != ""))
        .with_columns(encode_target("tid").alias("tid"))
    )
