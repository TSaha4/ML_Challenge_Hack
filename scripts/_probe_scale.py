"""Temporary probe: measure blocker index cost/throughput to size the audit benchmark."""
import gc
import time
import tracemalloc

import psutil

from src.blocking import AdaptiveBlocker
from src.features import extract_pairwise_features
from src.inference import normalize_rows
from src.pipeline import compute_corpus_token_idf, stream_tsv_chunks

TRAIN = "student_resource/dataset/train"
proc = psutil.Process()


def mb(x):
    return round(x / 1e6, 1)


print("building IDF (20k S1 names, as Phase 10 does)...")
t0 = time.time()
idf = compute_corpus_token_idf(f"{TRAIN}/train_source1.tsv", sample_size=20_000)
print(f"  idf tokens={len(idf):,} in {time.time()-t0:.1f}s")

# --- normalization throughput -------------------------------------------------
t0 = time.time()
n = 0
for chunk in stream_tsv_chunks(f"{TRAIN}/train_source2.tsv", chunk_size=50_000, n_rows=50_000):
    recs = normalize_rows(chunk)
    n += len(recs)
print(f"normalize_rows: {n:,} recs in {time.time()-t0:.1f}s -> {n/(time.time()-t0):,.0f} rec/s")

# --- index cost --------------------------------------------------------------
POOL = 200_000
blocker = AdaptiveBlocker(max_block_size=500, max_candidates_per_entity=100)
base_rss = proc.memory_info().rss
gc.collect()
tracemalloc.start()
t0 = time.time()
n = 0
for chunk in stream_tsv_chunks(f"{TRAIN}/train_source2.tsv", chunk_size=50_000, n_rows=POOL):
    recs = normalize_rows(chunk)
    blocker.fit(recs, idf_dict=idf)
    n += len(recs)
    del recs
elapsed = time.time() - t0
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nindex: {n:,} targets in {elapsed:.1f}s -> {n/elapsed:,.0f} rec/s (incl. normalization)")
print(f"traced index bytes: {mb(peak)} MB -> {peak/n:.0f} B/record -> 10.32M targets = {mb(peak/n*10_320_219)} MB")
print(f"keys: {len(blocker.index):,} ({len(blocker.index)/n:.2f} keys/record) | rss delta {mb(proc.memory_info().rss-base_rss)} MB")

# --- query + feature throughput ---------------------------------------------
s1_rows = []
for chunk in stream_tsv_chunks(f"{TRAIN}/train_source1.tsv", chunk_size=500, n_rows=500):
    s1_rows = normalize_rows(chunk)
t0 = time.time()
n_cands = 0
n_pairs = 0
t_feat = 0.0
for rec in s1_rows:
    hits = blocker.query(rec, idf_dict=idf)
    n_cands += len(hits)
    for h in hits:
        t1 = time.time()
        extract_pairwise_features(rec, rec, meta={"channel_count": h["channel_count"], "min_rank": h["min_rank"]}, idf_dict=idf)
        t_feat += time.time() - t1
        n_pairs += 1
qtime = time.time() - t0 - t_feat
print(f"\nS1 query+features: {len(s1_rows)} S1 -> {n_cands:,} candidates ({n_cands/len(s1_rows):.1f}/S1)")
print(f"  feature extraction: {n_pairs:,} pairs in {t_feat:.2f}s -> {n_pairs/max(1e-9,t_feat):,.0f} pairs/s, {t_feat/max(1,n_pairs)*1e6:.1f} us/pair")
print(f"  blocking query time: {qtime:.2f}s ({qtime/len(s1_rows)*1000:.2f} ms/S1)")
