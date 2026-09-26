import sys, time, gc, threading
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import psutil, pandas as pd, duckdb
from src.blocking import extract_blocking_keys, _BLOCK_MATCH_SQL, _BLOCK_BATCH_SQL
from src.training import iter_s1_batches
from src.pipeline import compute_corpus_token_idf

proc = psutil.Process()
peak = {'v': 0}; done = False
def samp():
    while not done:
        peak['v'] = max(peak['v'], proc.memory_info().rss); time.sleep(0.05)
threading.Thread(target=samp, daemon=True).start()
rss = lambda: round(proc.memory_info().rss/1e6)

idf = compute_corpus_token_idf('student_resource/dataset/train/train_source1.tsv', sample_size=20_000)
s1 = next(iter(iter_s1_batches('student_resource/dataset/train', s1_chunk_size=20_000,
                               max_s1=20_000, s1_offset=1_000_000)))
print(f's1={len(s1)} rss={rss()}', flush=True)

def make_frame(batch):
    rows = []
    for rec in batch:
        for co, (ch, val) in enumerate(extract_blocking_keys(rec, idf)):
            rows.append((rec['entity_id'], co, ch, val))
    f = pd.DataFrame(rows, columns=['s1_id','chan_order','channel','key'])
    return f.drop_duplicates(subset=['s1_id','channel','key'], keep='first')

frames = {cs: make_frame(s1[:cs]) for cs in (20_000, 10_000, 5_000)}
print('frames:', {k: len(v) for k, v in frames.items()}, 'rss=', rss(), flush=True)
del s1; gc.collect()

def attempt(cs, variant='real'):
    peak['v'] = 0
    con = duckdb.connect('artifacts/checkpoints/bench_blocking.duckdb', read_only=True)
    con.execute("PRAGMA memory_limit='1024MB'"); con.execute('PRAGMA threads=4')
    con.execute("PRAGMA temp_directory='artifacts/checkpoints'")
    con.register('query_keys', frames[cs])
    t0 = time.time()
    try:
        con.execute(_BLOCK_MATCH_SQL)
        t1 = time.time()
    except Exception as e:
        con.close()
        print(f'  cs={cs:>6} {variant:8s} FAIL@step1 {type(e).__name__}: {str(e)[:100]}', flush=True)
        return
    # window-only probe
    try:
        n = con.execute("SELECT count(*) FROM (SELECT row_number() OVER (PARTITION BY channel,key ORDER BY seq) AS rk FROM _block_matched)").fetchone()[0]
        t2 = time.time()
    except Exception as e:
        con.close()
        print(f'  cs={cs:>6} {variant:8s} FAIL@window ({time.time()-t1:.1f}s) {str(e)[:90]}', flush=True)
        return
    if variant == 'no_count':
        sql = _BLOCK_BATCH_SQL.replace('count(DISTINCT k.channel)           AS channel_count,', "string_agg(DISTINCT k.channel, '|') AS channels_x,")
    else:
        sql = _BLOCK_BATCH_SQL
    try:
        rows = con.execute(sql, [500]).fetchall()
        t3 = time.time()
        print(f'  cs={cs:>6} {variant:8s} OK rows={len(rows):>8} | step1={t1-t0:.1f}s win={t2-t1:.1f}s agg={t3-t2:.1f}s | peakRSS={round(peak["v"]/1e6)}MB', flush=True)
    except Exception as e:
        print(f'  cs={cs:>6} {variant:8s} FAIL@aggregate ({time.time()-t2:.1f}s after win ok) {str(e)[:110]} | peakRSS={round(peak["v"]/1e6)}MB', flush=True)
    finally:
        con.close()
    gc.collect()

print('--- attempts (window isolated, then full step2) ---', flush=True)
attempt(20_000)
attempt(10_000)
attempt(5_000)
attempt(10_000, variant='no_count')
attempt(5_000, variant='no_count')
print('done rss=', rss(), flush=True)
