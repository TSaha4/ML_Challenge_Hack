"""
Memory-safe, leak-free training-dataset construction and model fitting.

Implements the *real* Phase 12-15 pipeline of the notebook:

    12. training dataset from the full training split
        (ground truth -> adaptive blocking -> candidate generation ->
         positive recovery -> hard-negative sampling -> pairwise features);
    13. entity-aware train/validation split (no Source-1 entity on both sides);
    14. LightGBM training + Macro F_0.5 threshold sweep on the *validation* side;
    15. iterative hard-negative mining driven by a fitted baseline model.

Design constraints (measured in ``scripts/bench_train_scale.py``):

* the training split holds 2,206,821 Source-1 entities, 10,320,219 target records
  and 7,638,365 ground-truth pairs. Nothing that scales with the *full* candidate
  space (up to 100 candidates x 2.2M entities) may be materialised in RAM, so
  candidates are streamed per Source-1 batch and only the labelled sample
  (all recovered positives + a bounded number of negatives per entity) is kept;
* the inverted index lives on disk (``DuckDBBlocker``) because the in-memory index
  costs ~800 bytes per indexed target (~8 GB at this scale);
* normalised target records live in DuckDB and are fetched back only for the
  candidates needed by the current batch (same strategy as ``src.inference``);
* ground-truth labels live in DuckDB and are joined per batch.

Leakage guards enforced here:

* features never use the ground truth - labels are joined only *after* the feature
  vector has been computed;
* the F_0.5 threshold is tuned on validation entities only, and validation
  entities are never used for fitting;
* IDF weights are estimated from Source-1 training names only.
"""

from __future__ import annotations

import collections
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.blocking import AdaptiveBlocker, DuckDBBlocker
from src.features import extract_pairwise_features
from src.inference import TARGET_COLUMNS, normalize_rows, save_model_bundle
from src.metrics import evaluate_macro_f_beta
from src.pipeline import stream_tsv_chunks

#: Columns that identify a pair and are therefore never model features.
ID_COLUMNS: Tuple[str, str] = ("s1_id", "cand_id")

#: Feature columns produced by ``extract_pairwise_features`` (schema check helper).
EXPECTED_FEATURES: List[str] = [
    "exact_name_norm", "exact_name_core", "exact_country", "exact_postal",
    "name_jw", "name_core_jw", "name_fuzz_ratio", "name_fuzz_partial",
    "name_fuzz_token_sort", "name_fuzz_token_set", "name_translit_jw",
    "name_token_jaccard", "name_token_len_diff", "name_char_len_diff",
    "name_initials_match", "addr_jw", "addr_fuzz_ratio", "addr_token_sort",
    "addr_token_jaccard", "addr_char_len_diff",
    "name_idf_sum", "name_idf_max", "name_idf_weighted_jaccard",
    "meta_channel_count", "meta_min_rank",
]


class GroundTruthStore:
    """Disk-backed ground-truth lookup: Source-1 id -> set of matched target ids."""

    def __init__(self, db_path, memory_limit: str = "512MB", threads: int = 2):
        import duckdb

        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path)
        # Cap DuckDB's buffer pool: the default (80% of system RAM) let RSS
        # balloon to ~7 GB during the scale benchmark on a 14 GB machine.
        self.con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        self.con.execute(f"PRAGMA threads={int(threads)}")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS gt (source1_entity_id VARCHAR, matched_entity_ids VARCHAR)"
        )
        self._loaded = self.con.execute("SELECT count(*) FROM gt").fetchone()[0] > 0

    def load(self, gt_path, chunk_size: int = 500_000) -> int:
        """Load ``train_ground_truth.tsv`` once (singleton rows carry an empty string)."""
        if self._loaded:
            return 0
        n = 0
        for chunk in stream_tsv_chunks(str(gt_path), chunk_size=chunk_size):
            frame = chunk.select(["source1_entity_id", "matched_entity_ids"]).fill_null("")
            self.con.register("gt_chunk", frame.to_arrow())
            try:
                self.con.execute(
                    "INSERT INTO gt (source1_entity_id, matched_entity_ids) "
                    "SELECT source1_entity_id, matched_entity_ids FROM gt_chunk"
                )
            finally:
                self.con.unregister("gt_chunk")
            n += len(frame)
        self._loaded = True
        return n

    def fetch(self, s1_ids: Sequence[str]) -> Dict[str, Set[str]]:
        """Ground-truth sets for a batch of Source-1 ids (empty set = true singleton)."""
        if not s1_ids:
            return {}
        ids = pd.DataFrame({"id": list(dict.fromkeys(s1_ids))})
        self.con.register("gt_ids", ids)
        try:
            rows = self.con.execute(
                "SELECT g.source1_entity_id, g.matched_entity_ids FROM gt g "
                "JOIN gt_ids i ON g.source1_entity_id = i.id"
            ).fetchall()
        finally:
            self.con.unregister("gt_ids")
        return {
            s1_id: {tok.strip() for tok in str(matched or "").split(",") if tok.strip()}
            for s1_id, matched in rows
        }

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


class TargetRecordStore:
    """DuckDB store holding normalised Source-2/3 records (schema = TARGET_COLUMNS)."""

    def __init__(self, db_path, reset: bool = True, memory_limit: str = "768MB", threads: int = 2):
        import duckdb

        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path)
        # Cap DuckDB's buffer pool: the default (80% of system RAM) let RSS
        # balloon to ~7 GB during the scale benchmark on a 14 GB machine.
        self.con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        self.con.execute(f"PRAGMA threads={int(threads)}")
        if reset:
            self.con.execute("DROP TABLE IF EXISTS target_records")
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS target_records (
                   entity_id VARCHAR PRIMARY KEY,
                   country VARCHAR,
                   name_norm VARCHAR,
                   name_core VARCHAR,
                   name_translit VARCHAR,
                   name_initials VARCHAR,
                   addr_postal VARCHAR,
                   addr_norm VARCHAR
               )"""
        )

    def add(self, records: List[Dict[str, Any]]):
        if not records:
            return
        frame = pd.DataFrame(records, columns=TARGET_COLUMNS)
        self.con.register("target_chunk", frame)
        columns_sql = ", ".join(TARGET_COLUMNS)
        try:
            self.con.execute(
                f"INSERT OR REPLACE INTO target_records ({columns_sql}) "
                f"SELECT {columns_sql} FROM target_chunk"
            )
        finally:
            self.con.unregister("target_chunk")
        del frame

    def fetch(self, needed_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
        if not needed_ids:
            return {}
        ids = pd.DataFrame({"id": list(needed_ids)})
        self.con.register("needed_ids", ids)
        columns_sql = ", ".join(TARGET_COLUMNS)
        try:
            frame = self.con.execute(
                f"SELECT {columns_sql} FROM target_records t "
                "JOIN needed_ids n ON t.entity_id = n.id"
            ).fetchdf()
        finally:
            self.con.unregister("needed_ids")
        lookup = frame.set_index("entity_id").to_dict("index")
        del frame
        return lookup

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Blocking / candidate generation (shared by dataset build and HN mining)
# ---------------------------------------------------------------------------
def index_targets(
    train_dir,
    idf_dict: Optional[Dict[str, float]],
    target_store: TargetRecordStore,
    blocker,
    chunk_size: int = 100_000,
    target_limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Normalise and index ``train_source2.tsv`` + ``train_source3.tsv``.

    Each target record is normalised once and written to both the blocking index
    and the DuckDB record store. ``target_limit`` caps the records taken from each
    source file (audit/benchmark mode only; the real run indexes everything).
    """
    train_dir = Path(train_dir)
    counts: Dict[str, int] = {}
    total = 0
    for source_num in (2, 3):
        path = train_dir / f"train_source{source_num}.tsv"
        taken = 0
        started = time.time()
        for chunk in stream_tsv_chunks(str(path), chunk_size=chunk_size, n_rows=target_limit):
            records = normalize_rows(chunk)
            if not records:
                continue
            blocker.fit(records, idf_dict=idf_dict)
            target_store.add(records)
            taken += len(records)
            del records
            gc.collect()
        counts[path.name] = taken
        total += taken
        if verbose:
            print(f"  indexed {path.name}: {taken:,} records ({time.time() - started:.0f}s)")
    counts["total"] = total
    if hasattr(blocker, "finalize"):
        blocker.finalize()
    return counts


def iter_s1_batches(
    train_dir,
    s1_chunk_size: int = 20_000,
    max_s1: Optional[int] = None,
    s1_offset: int = 0,
) -> Iterator[List[Dict[str, Any]]]:
    """Stream normalised Source-1 training records in bounded batches."""
    path = Path(train_dir) / "train_source1.tsv"
    n_rows = None
    if max_s1 is not None:
        n_rows = s1_offset + max_s1
    skip = s1_offset
    produced = 0
    for chunk in stream_tsv_chunks(str(path), chunk_size=s1_chunk_size, n_rows=n_rows):
        if skip:
            if len(chunk) <= skip:
                skip -= len(chunk)
                continue
            chunk = chunk.slice(skip, len(chunk) - skip)
            skip = 0
        records = normalize_rows(chunk)
        del chunk
        if not records:
            continue
        if max_s1 is not None and produced + len(records) > max_s1:
            records = records[: max_s1 - produced]
        produced += len(records)
        yield records
        if max_s1 is not None and produced >= max_s1:
            break


def build_training_dataset(
    train_dir,
    idf_dict: Optional[Dict[str, float]],
    blocker,
    target_store: TargetRecordStore,
    gt_store: GroundTruthStore,
    *,
    max_s1: Optional[int] = None,
    s1_offset: int = 0,
    neg_ratio: int = 2,
    singleton_neg_ratio: int = 1,
    s1_chunk_size: int = 20_000,
    out_parquet: Optional[str] = None,
    return_frame: bool = True,
    rng_seed: int = 42,
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Dict[str, Any]]:
    """
    Build the labelled pairwise training sample from the training split.

    For every streamed Source-1 entity the candidates are retrieved from the
    blocking index, labelled against ground truth and reduced to

    * **all recoverable positives** (ground-truth pairs that blocking retrieved);
    * up to ``neg_ratio * positives`` negatives per entity, drawn half from the top
      of the blocking order (hard negatives - many channels / small rank) and half
      uniformly from the remaining pool (representative negatives);
    * for true singletons (empty ground truth) up to ``singleton_neg_ratio``
      negatives, so the model learns the singleton behaviour Macro F_0.5 rewards.

    Ground-truth pairs that blocking does *not* retrieve are counted in the returned
    statistics (``positive_recall``): they are unrecoverable, and this is the
    headline number to check before trusting a trained model.

    Returns ``(X, y, stats)``; ``X`` is ``None`` when ``return_frame=False``
    (features are then written incrementally to ``out_parquet``).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(rng_seed)
    stats: Dict[str, Any] = collections.Counter()
    processed_s1_ids: List[str] = []
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    writer = None
    parquet_path = Path(out_parquet) if out_parquet else None
    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if parquet_path.exists():
            parquet_path.unlink()

    started = time.time()
    batch_idx = 0
    try:
        for s1_records in iter_s1_batches(train_dir, s1_chunk_size, max_s1, s1_offset):
            batch_idx += 1
            ids = [rec["entity_id"] for rec in s1_records]
            hits = blocker.query_batch(s1_records, idf_dict=idf_dict)
            needed = {c["candidate_id"] for cands in hits.values() for c in cands}
            lookup = target_store.fetch(needed)
            gt_map = gt_store.fetch(ids)
            del needed

            for rec in s1_records:
                s1_id = rec["entity_id"]
                truth = gt_map.get(s1_id, set())
                cands = hits.get(s1_id, [])
                stats["s1_entities"] += 1
                stats["gt_pairs"] += len(truth)
                stats["candidates"] += len(cands)
                if not cands:
                    stats["s1_without_candidates"] += 1
                positives = [c for c in cands if c["candidate_id"] in truth]
                neg_pool = [c for c in cands if c["candidate_id"] not in truth]
                stats["positive_hits"] += len(positives)
                stats["negative_candidates"] += len(neg_pool)

                if truth:
                    quota = min(len(neg_pool), neg_ratio * max(1, len(positives)))
                    n_hard = (quota + 1) // 2
                    hard = neg_pool[:n_hard]
                    rest = neg_pool[n_hard:]
                    if quota - n_hard > 0 and rest:
                        take = min(quota - n_hard, len(rest))
                        picked = rng.choice(len(rest), size=take, replace=False)
                        sample = [rest[int(i)] for i in np.sort(picked)]
                    else:
                        sample = []
                    keep = positives + hard + sample
                else:
                    keep = neg_pool[:singleton_neg_ratio]

                for cand in keep:
                    cand_rec = lookup.get(cand["candidate_id"])
                    if cand_rec is None:
                        stats["pairs_skipped_unindexed"] += 1
                        continue
                    feats = extract_pairwise_features(
                        rec,
                        cand_rec,
                        meta={
                            "channel_count": cand["channel_count"],
                            "min_rank": cand["min_rank"],
                        },
                        idf_dict=idf_dict,
                    )
                    feats["s1_id"] = s1_id
                    feats["cand_id"] = cand["candidate_id"]
                    feature_rows.append(feats)
                    labels.append(1 if cand["candidate_id"] in truth else 0)

            processed_s1_ids.extend(ids)
            del hits, lookup, gt_map

            if parquet_path is not None and feature_rows:
                batch_frame = pd.DataFrame(feature_rows)
                batch_frame["label"] = np.asarray(labels, dtype=np.int8)
                table = pa.Table.from_pandas(batch_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(parquet_path), table.schema)
                writer.write_table(table)
                feature_rows = []
                labels = []
                del table, batch_frame
            del s1_records
            gc.collect()

            if verbose and batch_idx % 5 == 0:
                print(
                    f"  batch {batch_idx}: {stats['s1_entities']:,} S1 | "
                    f"{stats['candidates']:,} candidates | "
                    f"{stats['positive_hits']:,} positives recovered | "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    if parquet_path is not None:
        if return_frame:
            frame = pd.read_parquet(parquet_path)
            y = frame.pop("label").to_numpy(dtype=np.int8) if "label" in frame.columns else None
            for col in frame.columns:
                if frame[col].dtype == np.float64:
                    frame[col] = frame[col].astype(np.float32)
        else:
            frame, y = None, None
    else:
        frame = pd.DataFrame(feature_rows) if (return_frame and feature_rows) else None
        y = np.asarray(labels, dtype=np.int8) if (return_frame and labels) else None
    del feature_rows, labels
    gc.collect()

    n_pos = stats["positive_hits"]
    summary: Dict[str, Any] = {
        "s1_entities": int(stats["s1_entities"]),
        "gt_pairs": int(stats["gt_pairs"]),
        "positive_hits": int(n_pos),
        "positive_recall": round(n_pos / stats["gt_pairs"], 4) if stats["gt_pairs"] else 0.0,
        "positive_missed": int(stats["gt_pairs"] - n_pos),
        "candidates": int(stats["candidates"]),
        "candidates_per_s1": round(stats["candidates"] / max(1, stats["s1_entities"]), 2),
        "negative_candidates": int(stats["negative_candidates"]),
        "s1_without_candidates": int(stats["s1_without_candidates"]),
        "pairs_skipped_unindexed": int(stats["pairs_skipped_unindexed"]),
        "rows": int(len(frame)) if frame is not None else -1,
        "positives_kept": int(y.sum()) if y is not None else -1,
        "negatives_kept": int(len(y) - y.sum()) if y is not None else -1,
        "processed_s1_ids": processed_s1_ids,
        "parquet": str(parquet_path) if parquet_path else None,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if frame is not None:
        summary["feature_columns"] = [c for c in frame.columns if c not in ID_COLUMNS]
    return frame, y, summary


# ---------------------------------------------------------------------------
# Phase 13 - entity-aware split (anti-leakage)
# ---------------------------------------------------------------------------
def entity_aware_split(
    processed_s1_ids: Sequence[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split Source-1 *entities* (not rows) into train/validation parts.

    Every pair of an entity lands on one side only, so a validation entity can
    never be memorised during fitting. Returns ``(train_ids, val_ids)``.
    """
    ids = np.asarray(list(dict.fromkeys(processed_s1_ids)))
    train_ids, val_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
    overlap = set(train_ids.tolist()) & set(val_ids.tolist())
    if overlap:  # pragma: no cover - defensive
        raise AssertionError(f"entity-aware split leaked {len(overlap)} Source-1 ids")
    return train_ids, val_ids


def split_masks(frame: pd.DataFrame, train_ids: Sequence[str], val_ids: Sequence[str]):
    """Row masks identifying the train/validation part of a feature frame."""
    train_mask = frame["s1_id"].isin(set(train_ids)).to_numpy()
    val_mask = frame["s1_id"].isin(set(val_ids)).to_numpy()
    overlap = int((train_mask & val_mask).sum())
    if overlap:  # pragma: no cover - defensive
        raise AssertionError(f"{overlap} rows belong to both split sides")
    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Phase 14 - threshold tuning on held-out entities
# ---------------------------------------------------------------------------
def tune_threshold_f05(
    model,
    frame: pd.DataFrame,
    s1_ids: Sequence[str],
    gt_map: Dict[str, Set[str]],
    feature_cols: Optional[Sequence[str]] = None,
    thresholds: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Sweep decision thresholds and report Macro F_0.5 on the validation entities.

    ``gt_map`` must contain *all* validation Source-1 entities, singletons
    included (empty set = true singleton), because Macro F_0.5 averages over every
    entity. Predictions are built only from rows above the threshold, so an entity
    with no surviving pair counts as an empty prediction (exactly what the hidden
    evaluation does).
    """
    feature_cols = list(feature_cols or [c for c in frame.columns if c not in ID_COLUMNS])
    probabilities = model.predict_proba(frame[feature_cols])[:, 1]
    cand_ids = frame["cand_id"].to_numpy()

    pairs_per_entity: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for s1_id, cand_id, prob in zip(s1_ids, cand_ids, probabilities):
        pairs_per_entity[s1_id].append((cand_id, float(prob)))

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)

    rows = []
    best_threshold, best_score = 0.5, -1.0
    best_metrics: Dict[str, float] = {}
    for threshold in thresholds:
        predictions = {
            s1_id: {cand_id for cand_id, prob in pairs if prob >= threshold}
            for s1_id, pairs in pairs_per_entity.items()
        }
        metrics = evaluate_macro_f_beta(gt_map, predictions, beta=0.5)
        rows.append({
            "threshold": float(threshold),
            "macro_f05": round(metrics["macro_f_beta"], 4),
            "singleton_f05": round(metrics["singleton_f_beta"], 4),
            "matched_f05": round(metrics["matched_f_beta"], 4),
            "n_matches": sum(len(v) for v in predictions.values()),
        })
        if metrics["macro_f_beta"] > best_score:
            best_score, best_threshold, best_metrics = (
                metrics["macro_f_beta"], float(threshold), metrics,
            )

    if verbose:
        print(f"Optimal validation threshold: {best_threshold} "
              f"(Macro F_0.5 = {best_score:.4f})")
    return {
        "best_threshold": best_threshold,
        "best_macro_f05": float(best_score),
        "best_metrics": {k: float(v) for k, v in best_metrics.items()},
        "sweep": rows,
        "n_val_entities": len(gt_map),
        "n_val_singletons": int(sum(1 for v in gt_map.values() if not v)),
    }


# ---------------------------------------------------------------------------
# Phase 14 - LightGBM fitting
# ---------------------------------------------------------------------------
#: Default LightGBM parameters (same family as the notebook baseline, sized for a
#: dataset of millions of pairs instead of the 100-row prototype).
DEFAULT_LGB_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 4,
    "verbose": -1,
}


def train_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: Sequence[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 30,
    verbose: bool = True,
):
    """
    Fit the LightGBM classifier used for inference.

    Keeps the scikit-learn API (``predict_proba`` / ``feature_importances_``) so the
    saved bundle stays directly usable by ``python -m src.inference``. Early stopping
    and its metric report use the held-out (entity-aware) validation rows only.
    """
    import lightgbm as lgb

    merged = dict(DEFAULT_LGB_PARAMS)
    merged.update(params or {})
    model = lgb.LGBMClassifier(**merged)

    fit_kwargs: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val[list(feature_cols)], y_val)]
        fit_kwargs["eval_metric"] = "binary_logloss"
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=verbose),
            lgb.log_evaluation(period=25 if verbose else 0),
        ]
    model.fit(X_train[list(feature_cols)], y_train, **fit_kwargs)
    return model


# ---------------------------------------------------------------------------
# Phase 15 - streaming hard-negative mining
# ---------------------------------------------------------------------------
def mine_hard_negatives(
    train_dir,
    idf_dict: Optional[Dict[str, float]],
    blocker,
    target_store: TargetRecordStore,
    gt_store: GroundTruthStore,
    model,
    feature_cols: Sequence[str],
    *,
    max_s1: Optional[int] = None,
    s1_offset: int = 0,
    score_threshold: float = 0.55,
    s1_chunk_size: int = 5_000,
    keep_positives: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Score *every* blocking candidate of a Source-1 slice with a fitted model and keep
    the false positives it is most confident about (``prob >= score_threshold``).

    This is the real Phase 15 step: mined rows are appended to the training sample and
    the model is refit, which pushes the decision boundary up exactly where the
    baseline is over-confident (precision is worth 2x recall under F_0.5). Positives
    found in the mined slice are kept too when ``keep_positives`` is set.
    """
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    stats: Dict[str, Any] = collections.Counter()
    started = time.time()
    batch_idx = 0

    for s1_records in iter_s1_batches(train_dir, s1_chunk_size, max_s1, s1_offset):
        batch_idx += 1
        ids = [rec["entity_id"] for rec in s1_records]
        hits = blocker.query_batch(s1_records, idf_dict=idf_dict)
        needed = {c["candidate_id"] for cands in hits.values() for c in cands}
        lookup = target_store.fetch(needed)
        gt_map = gt_store.fetch(ids)
        del needed

        batch_rows: List[Dict[str, float]] = []
        batch_truth: List[bool] = []
        for rec in s1_records:
            s1_id = rec["entity_id"]
            truth = gt_map.get(s1_id, set())
            for cand in hits.get(s1_id, []):
                cand_rec = lookup.get(cand["candidate_id"])
                if cand_rec is None:
                    continue
                feats = extract_pairwise_features(
                    rec,
                    cand_rec,
                    meta={"channel_count": cand["channel_count"], "min_rank": cand["min_rank"]},
                    idf_dict=idf_dict,
                )
                feats["s1_id"] = s1_id
                feats["cand_id"] = cand["candidate_id"]
                batch_rows.append(feats)
                batch_truth.append(cand["candidate_id"] in truth)

        stats["pairs_scored"] += len(batch_rows)
        if batch_rows:
            batch_frame = pd.DataFrame(batch_rows)
            probabilities = model.predict_proba(batch_frame[list(feature_cols)])[:, 1]
            for row, is_positive, prob in zip(batch_rows, batch_truth, probabilities):
                if is_positive:
                    if keep_positives:
                        labels.append(1)
                        feature_rows.append(row)
                        stats["positives_kept"] += 1
                elif prob >= score_threshold:
                    labels.append(0)
                    feature_rows.append(row)
                    stats["hard_negatives_kept"] += 1
            del batch_frame, probabilities
        del batch_rows, batch_truth, hits, lookup, gt_map, s1_records
        gc.collect()

        if verbose and batch_idx % 4 == 0:
            print(
                f"  mined {stats['pairs_scored']:,} pairs | "
                f"{stats['hard_negatives_kept']:,} hard negatives | "
                f"{time.time() - started:.0f}s",
                flush=True,
            )

    frame = pd.DataFrame(feature_rows)
    y = np.asarray(labels, dtype=np.int8)
    summary = {
        "pairs_scored": int(stats["pairs_scored"]),
        "hard_negatives_kept": int(stats["hard_negatives_kept"]),
        "positives_kept": int(stats["positives_kept"]),
        "score_threshold": float(score_threshold),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return frame, y, summary


def save_json(path, payload: Dict[str, Any]) -> str:
    """Write a JSON artefact, converting numpy scalars/arrays to native types."""
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2)
    return str(path)

