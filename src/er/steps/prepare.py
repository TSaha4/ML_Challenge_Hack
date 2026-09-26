"""Step 1: learn transliteration maps and normalise a split to compact parquet.

    python -m src.er.steps.prepare --split train
    python -m src.er.steps.prepare --split test

Train uses maps learned on the training fold only (validation Source-1 entities,
``s1 % 4 == 0``, are held out) so validation scores stay honest; test uses maps
learned on all training pairs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.er.io import DATA_DIR, load_ground_truth, load_s1, load_targets
from src.er.text import normalize_frame
from src.er.translit import learn_addr_map, learn_name_map, load_maps, save_maps

ART = Path("artifacts/er")
VAL_MOD = 4


def learn_maps(data_dir: Path) -> None:
    gt = load_ground_truth(data_dir)
    s1 = load_s1(data_dir, "train").select(
        pl.col("id").alias("s1"), pl.col("business_name").alias("sname"),
        pl.col("business_address").alias("saddr"),
    )
    tg = load_targets(data_dir, "train").filter(
        pl.col("business_name").str.contains(r"[ऀ-෿]")
        | pl.col("business_address").str.contains(r"[ऀ-෿]")
    ).select(pl.col("id").alias("tid"), pl.col("business_name").alias("tname"),
             pl.col("business_address").alias("taddr"))
    pairs = tg.join(gt, on="tid").join(s1, on="s1")
    for tag, sub in (("trainfold", pairs.filter(pl.col("s1") % VAL_MOD != 0)), ("all", pairs)):
        nm, am = learn_name_map(sub), learn_addr_map(sub)
        save_maps(ART / f"translit_{tag}.json", nm, am)
        print(f"  translit[{tag}]: {len(nm):,} name tokens, {len(am):,} address tokens")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--slice", type=int, default=1_000_000)
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    maps_file = ART / ("translit_trainfold.json" if args.split == "train" else "translit_all.json")
    if not maps_file.exists():
        print("learning transliteration maps ...")
        learn_maps(data_dir)
    name_map, addr_map = load_maps(maps_file)

    for side, loader in (("s1", load_s1), ("tg", load_targets)):
        df = loader(data_dir, args.split)
        # slice so the transient string/list columns stay small
        norm = pl.concat([
            normalize_frame(df.slice(i, args.slice), name_map, addr_map)
            for i in range(0, df.height, args.slice)
        ])
        del df
        if side == "tg":
            norm = norm.with_columns((pl.col("id") // 10_000_000_000).cast(pl.Int8).alias("src"))
        out = ART / f"{args.split}_{side}.parquet"
        norm.write_parquet(out, compression="zstd")
        print(f"  {args.split}/{side}: {norm.height:,} rows -> {out} "
              f"({out.stat().st_size / 1e6:.0f} MB)  {time.time() - t0:.0f}s", flush=True)
        del norm


if __name__ == "__main__":
    main()
