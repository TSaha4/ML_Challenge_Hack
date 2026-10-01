"""GPU gradient-boosted matcher, target->Source-1 assignment and macro-F0.5 tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import numpy as np
import polars as pl
from src.er.artifacts import FEATURE_VERSION, write_json

from src.er.gpu_features import FEATURES

if TYPE_CHECKING:
    import xgboost as xgb

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 9,
    "eta": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "max_bin": 256,
    "nthread": 2,
}


def to_matrix(feats: pl.DataFrame, names=None) -> np.ndarray:
    return feats.select([pl.col(c).cast(pl.Float32) for c in (names or FEATURES)]).to_numpy()


def train(train_df: pl.DataFrame, valid_df: pl.DataFrame, rounds: int = 1500,
          early_stop: int = 50) -> xgb.Booster:
    import xgboost as xgb
    dtr = xgb.DMatrix(to_matrix(train_df), label=train_df["y"].to_numpy(), feature_names=FEATURES)
    dva = xgb.DMatrix(to_matrix(valid_df), label=valid_df["y"].to_numpy(), feature_names=FEATURES)
    return xgb.train(XGB_PARAMS, dtr, rounds, evals=[(dtr, "train"), (dva, "valid")],
                     early_stopping_rounds=early_stop, verbose_eval=100)


def predict(booster: xgb.Booster, feats: pl.DataFrame) -> np.ndarray:
    import xgboost as xgb
    names = booster.feature_names or FEATURES  # the model's own feature list
    d = xgb.DMatrix(to_matrix(feats, names), feature_names=names)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


def assign(scored: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Each target goes to its highest-probability Source-1 when ``p >= threshold``."""
    best = scored.sort(["p", "s1"], descending=[True, False]).unique("tid", keep="first", maintain_order=True)
    return best.filter(pl.col("p") >= threshold).select("tid", "s1", "p")


def assign_supported(scored: pl.DataFrame, thr: float, thr_support: float,
                     conf: float = 0.95) -> pl.DataFrame:
    """Argmax assignment with sibling support.

    A target is kept when ``p >= thr``, or when ``p >= thr_support`` and its
    Source-1 entity already has at least one *other* target assigned with
    ``p >= conf`` (records of the same business corroborate each other).
    """
    best = scored.sort(["p", "s1"], descending=[True, False]).unique("tid", keep="first", maintain_order=True).select("tid", "s1", "p")
    conf_cnt = best.group_by("s1").agg((pl.col("p") >= conf).sum().alias("_nconf"))
    best = best.join(conf_cnt, on="s1", how="left").with_columns(
        (pl.col("_nconf") - (pl.col("p") >= conf).cast(pl.UInt32)).alias("_others"))
    keep = (pl.col("p") >= thr) | ((pl.col("p") >= thr_support) & (pl.col("_others") >= 1))
    return best.filter(keep).select("tid", "s1", "p")


def tune_supported(scored: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series,
                   thr_grid=None, sup_grid=None, conf: float = 0.95):
    """Grid-search ``(thr, thr_support)`` for :func:`assign_supported` on macro F0.5."""
    thr_grid = list(thr_grid or np.round(np.arange(0.60, 0.925, 0.025), 3))
    sup_grid = list(sup_grid or np.round(np.arange(0.05, 0.80, 0.05), 3))
    rows = []
    for t in thr_grid:
        for s in sup_grid:
            if s > t:
                continue
            m = macro_f05(assign_supported(scored, t, s, conf), truth, universe)
            rows.append({**m, "threshold": float(t), "thr_support": float(s)})
    return max(rows, key=lambda r: r["macro_f05"]), rows


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series, beta: float = 0.5) -> Dict[str, float]:
    """Official macro F_beta over ``universe`` Source-1 ids (singletons included).

    ``pred``/``truth`` are long ``(s1, tid)`` pair tables.
    """
    u = pl.DataFrame({"s1": universe})
    p = pred.join(u, on="s1", how="semi").select("s1", "tid")
    t = truth.join(u, on="s1", how="semi").select("s1", "tid")
    tp = p.join(t, on=["s1", "tid"]).group_by("s1").len("tp")
    npred = p.group_by("s1").len("np")
    ntrue = t.group_by("s1").len("nt")
    df = (u.join(tp, on="s1", how="left").join(npred, on="s1", how="left")
          .join(ntrue, on="s1", how="left").fill_null(0))
    b2 = beta * beta
    prec = pl.col("tp") / pl.col("np").clip(1)
    rec = pl.col("tp") / pl.col("nt").clip(1)
    f = (
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise((1 + b2) * prec * rec / (b2 * prec + rec))
    )
    df = df.with_columns(f.alias("f"))
    single = df.filter(pl.col("nt") == 0)
    multi = df.filter(pl.col("nt") > 0)
    tp_all, np_all, nt_all = df["tp"].sum(), df["np"].sum(), df["nt"].sum()
    return {
        "macro_f05": float(df["f"].mean()),
        "singleton_f05": float(single["f"].mean()) if single.height else float("nan"),
        "matched_f05": float(multi["f"].mean()) if multi.height else float("nan"),
        "pair_precision": tp_all / max(np_all, 1),
        "pair_recall": tp_all / max(nt_all, 1),
        "n_s1": df.height,
    }


def tune_threshold(scored: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series,
                   grid: Optional[Iterable[float]] = None) -> tuple[float, List[dict]]:
    """Exact macro-F0.5 sweep after global target assignment, without repeated sorts.

    Each winner adds a delta to its entity's F score. Equal probabilities enter
    together. The returned curve is a bounded diagnostic sample plus the optimum.
    Highest threshold wins ties, including the predict-nothing endpoint.
    """
    if universe.len() == 0:
        raise ValueError("Cannot calibrate an empty entity universe")
    best = assign(scored, float('-inf'))
    if not np.isfinite(best['p'].to_numpy()).all():
        raise ValueError("Non-finite model scores")
    if grid is not None:
        rows = [{**macro_f05(best.filter(pl.col('p') >= t), truth, universe),
                 'threshold': float(t)} for t in grid]
        if not rows:
            raise ValueError("Empty threshold grid")
        row = max(rows, key=lambda r: (r['macro_f05'], r['threshold']))
        return row['threshold'], rows
    u = pl.DataFrame({'s1': universe}).unique().with_row_index('_entity')
    nt = (u.join(truth.unique(['s1', 'tid']).group_by('s1').len('_nt'), on='s1', how='left')
          .fill_null(0).sort('_entity')['_nt'].to_numpy().astype(np.float64))
    w = (best.join(u, on='s1', how='inner')
         .join(truth.select('s1', 'tid').unique().with_columns(pl.lit(1).alias('_y')),
               on=['s1', 'tid'], how='left').with_columns(pl.col('_y').fill_null(0))
         .sort(['p', 's1', 'tid'], descending=[True, False, False]))
    endpoint = float(np.nextafter(max(1., float(best['p'].max() or 0)), np.inf))
    if not w.height:
        row = {**macro_f05(best.head(0), truth, universe), 'threshold': endpoint}
        return endpoint, [row]
    entity, y, probabilities = w['_entity'].to_numpy(), w['_y'].to_numpy(), w['p'].to_numpy()
    order = np.argsort(entity, kind='stable')
    e, yy = entity[order], y[order]
    starts = np.r_[True, e[1:] != e[:-1]]
    first = np.maximum.accumulate(np.where(starts, np.arange(len(e)), 0))
    count = np.arange(len(e)) - first + 1
    cumulative = np.cumsum(yy)
    tp = cumulative - np.where(first > 0, cumulative[np.maximum(first - 1, 0)], 0)
    after = 1.25 * tp / (.25 * nt[e] + count)
    before = np.r_[0., after[:-1]]
    before[starts] = (nt[e[starts]] == 0).astype(float)
    delta = np.empty(len(e), dtype=float)
    delta[order] = after - before
    boundaries = np.r_[0, np.flatnonzero(probabilities[1:] != probabilities[:-1]) + 1]
    totals = (float((nt == 0).sum()) + np.cumsum(np.add.reduceat(delta, boundaries))) / len(nt)
    # Include the empty prediction. np.argmax prefers the first/higher threshold.
    scores = np.r_[float((nt == 0).mean()), totals]
    thresholds = np.r_[endpoint, probabilities[boundaries]]
    optimum = int(np.argmax(scores))
    indices = np.unique(np.r_[np.linspace(0, len(scores)-1, min(101, len(scores)), dtype=int), optimum])
    # Full breakdown only for sampled points; assignment is already done.
    rows = [{**macro_f05(best.filter(pl.col('p') >= thresholds[i]), truth, universe),
             'threshold': float(thresholds[i])} for i in indices]
    return float(thresholds[optimum]), rows


def save(booster: xgb.Booster, meta: dict, model_dir: str | Path) -> None:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    booster.save_model(d / "xgb.json")
    meta = {**meta, "features": booster.feature_names or FEATURES,
            "feature_version": FEATURE_VERSION, "best_iteration": booster.best_iteration}
    write_json(d / "meta.json", meta)


def load(model_dir: str | Path) -> tuple[xgb.Booster, dict]:
    import xgboost as xgb
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model uses old feature semantics. Retrain before inference.")
    booster = xgb.Booster()
    booster.load_model(d / "xgb.json")
    booster.set_param({"device": "cuda", "nthread": 2})
    booster.best_iteration = meta["best_iteration"]
    return booster, meta
