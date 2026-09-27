"""Disjoint calibration/holdout folds, both excluded from learned transliteration."""
import polars as pl

FOLD_MOD = 100
CALIBRATION_FOLD = 0
HOLDOUT_FOLD = 4


def reserved_entities():
    return (pl.col('s1') % FOLD_MOD).is_in([CALIBRATION_FOLD, HOLDOUT_FOLD])


def reserved_targets(candidates: pl.LazyFrame, truth: pl.DataFrame):
    # Include positives missed by blocking: these must never train the neural model.
    return pl.concat([
        candidates.filter(reserved_entities()).select('tid').collect(),
        truth.filter(reserved_entities()).select('tid'),
    ]).unique()
