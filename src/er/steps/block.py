"""Step 2: GPU candidate generation for a split.

    python -m src.er.steps.block --split train
    python -m src.er.steps.block --split test

Writes ``artifacts/er/{split}_cands/part-NNN.parquet`` (tid, s1, sn, sa, nk, rk, qs, rq),
one part per 1M targets in file order, plus ``{split}_s1stats.parquet``.
Targets are streamed from parquet, so RAM stays ~2-3 GB; the index lives in VRAM.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.er.resources import lower_priority, rss_gb, wait_for_ram

ART = Path("artifacts/er")
COLS = ["id", "cty", "core", "sq", "ad"]
PART = 1_000_000


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--k", type=int, default=10, help="keep IDF top-k")
    ap.add_argument("--k-wide", type=int, default=200, help="IDF top-k re-ranked by char similarity")
    ap.add_argument("--k-char", type=int, default=6, help="also keep char-similarity top-k")
    args = ap.parse_args(argv)
    lower_priority()
    from src.er.features import s1_stats
    from src.er.gpu_blocking import GpuIndex

    t0 = time.time()
    s1 = pl.read_parquet(ART / f"{args.split}_s1.parquet", columns=COLS)
    index = GpuIndex(s1, cap=args.cap)
    del s1
    print(f"index: {index.n_rows:,} keys (cap={args.cap})  {time.time() - t0:.0f}s  rss {rss_gb():.1f}GB", flush=True)

    out = ART / f"{args.split}_cands"
    out.mkdir(exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    src = pl.scan_parquet(ART / f"{args.split}_tg.parquet").select(COLS)
    n_tg = src.select(pl.len()).collect().item()
    for i, start in enumerate(range(0, n_tg, PART)):
        wait_for_ram()
        tg = src.slice(start, PART).collect()
        part = index.candidates(tg, k=args.k, k_wide=args.k_wide, k_char=args.k_char)
        part.sort("tid", "rk").write_parquet(out / f"part-{i:03d}.parquet", compression="zstd")
        print(f"  part {i}: targets {start:,}+{tg.height:,} -> {part.height:,} pairs  "
              f"{time.time() - t0:.0f}s  rss {rss_gb():.1f}GB", flush=True)
        del tg, part
    del index
    cands = pl.scan_parquet(out / "part-*.parquet")
    s1_stats(cands).write_parquet(ART / f"{args.split}_s1stats.parquet")
    n = cands.select(pl.len()).collect().item()
    from src.er.siblings import build_sibling_tables
    build_sibling_tables(str(out / "part-*.parquet"), str(ART / f"{args.split}_tg.parquet"),
                         str(ART / f"{args.split}_s1.parquet"), ART / f"{args.split}_sibs")
    print(f"DONE {n:,} candidate pairs for {n_tg:,} targets -> {out}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
