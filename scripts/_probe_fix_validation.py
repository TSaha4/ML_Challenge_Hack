import sys, time, gc, threading
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import psutil
from src.blocking import DuckDBBlocker
from src.training import iter_s1_batches
from src.pipeline import compute_corpus_token_idf

proc = psutil.Process()
peak = {'v': 0}; done = False
def samp():
    while not done:
        peak['v'] = max(peak['v'], proc.memory_info().rss); time.sleep(0.05)
threading.Thread(target=samp, daemon=True).start()

print('building 20k S1 sample + idf...', flush=True)
idf = compute_corpus_token_idf('student_resource/dataset/train/train_source1.tsv', sample_size=20_000)
s1 = next(iter(iter_s1_batches('student_resource/dataset/train', s1_chunk_size=20_000,
                               max_s1=20_000, s1_offset=1_000_000)))
print(f's1={len(s1)} rss={round(proc.memory_info().rss/1e6)}MB', flush=True)

blocker = DuckDBBlocker('artifacts/checkpoints/bench_blocking.duckdb',
                        max_block_size=500, max_candidates_per_entity=100,
                        memory_limit='1024MB', threads=4,
                        temp_directory='artifacts/checkpoints')
t0 = time.time()
hits = blocker.query_batch(s1, idf_dict=idf)
dt = time.time() - t0
total = sum(len(v) for v in hits.values())
withcand = sum(1 for v in hits.values() if v)
done = True
print(f'OK: {len(hits)} s1 | {total} candidates ({total/len(s1):.1f}/s1) | '
      f'{withcand} s1 with cands | {dt:.1f}s | peakRSS={round(peak["v"]/1e6)}MB | '
      f'endRSS={round(proc.memory_info().rss/1e6)}MB', flush=True)
blocker.close(); gc.collect()
