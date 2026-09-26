"""GPU gradient-boosted matcher, target->Source-1 assignment and macro-F0.5 tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import polars as pl
import xgboost as xgb

from src.er.gpu_features import FEATURES

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


def to_matrix(feats: pl.DataFrame) -> np.ndarray:
    return feats.select([pl.col(c).cast(pl.Float32) for c in FEATURES]).to_numpy()


def train(train_df: pl.DataFrame, valid_df: pl.DataFrame, rounds: int = 1500,
          early_stop: int = 50) -> xgb.Booster:
    dtr = xgb.DMatrix(to_matrix(train_df), label=train_df["y"].to_numpy(), feature_names=FEATURES)
    dva = xgb.DMatrix(to_matrix(valid_df), label=valid_df["y"].to_numpy(), feature_names=FEATURES)
    return xgb.train(XGB_PARAMS, dtr, rounds, evals=[(dtr, "train"), (dva, "valid")],
                     early_stopping_rounds=early_stop, verbose_eval=100)


def predict(booster: xgb.Booster, feats: pl.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(to_matrix(feats), feature_names=FEATURES)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


def assign(scored: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Each target goes to its highest-probability Source-1 when ``p >= threshold``."""
    best = scored.sort("p", descending=True).unique("tid", keep="first")
    return best.filter(pl.col("p") >= threshold).select("tid", "s1", "p")


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
    grid = list(grid or np.round(np.arange(0.20, 0.96, 0.025), 3))
    rows = []
    for thr in grid:
        m = macro_f05(assign(scored, thr), truth, universe)
        m["threshold"] = float(thr)
        rows.append(m)
    best = max(rows, key=lambda r: r["macro_f05"])
    return best["threshold"], rows


def save(booster: xgb.Booster, meta: dict, model_dir: str | Path) -> None:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    booster.save_model(d / "xgb.json")
    meta = {**meta, "features": FEATURES, "best_iteration": booster.best_iteration}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load(model_dir: str | Path) -> tuple[xgb.Booster, dict]:
    d = Path(model_dir)
    booster = xgb.Booster()
    booster.load_model(d / "xgb.json")
    meta = json.loads((d / "meta.json").read_text())
    booster.set_param({"device": "cuda", "nthread": 2})
    booster.best_iteration = meta["best_iteration"]
    return booster, meta
