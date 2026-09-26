import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import psutil, pandas as pd, duckdb
from src.blocking import DuckDBBlocker, extract_blocking_keys
from src.inference import normalize_rows
from src.pipeline import stream_tsv_chunks
from src.training import iter_s1_batches

proc = psutil.Process()
rss = lambda: round(proc.memory_info().rss/1e6)
print('rss start:', rss())

# same 20k S1 sample as the scale stage (offset 1,000,000)
from src.pipeline import compute_corpus_token_idf
idf = compute_corpus_token_idf('student_resource/dataset/train/train_source1.tsv', sample_size=20_000)
s1 = next(iter(iter_s1_batches('student_resource/dataset/train', s1_chunk_size=20_000,
                               max_s1=20_000, s1_offset=1_000_000)))
key_rows = []
for rec in s1:
    for chan_order, (ch, val) in enumerate(extract_blocking_keys(rec, idf)):
        key_rows.append((rec['entity_id'], chan_order, ch, val))
frame = pd.DataFrame(key_rows, columns=['s1_id','chan_order','channel','key'])
frame = frame.drop_duplicates(subset=['s1_id','channel','key'], keep='first')
print(f's1={len(s1)} key_rows={len(frame)} distinct keys={len(frame[["channel","key"]].drop_duplicates())} rss={rss()}')

del s1, key_rows
gc.collect()

con = duckdb.connect('artifacts/checkpoints/bench_blocking.duckdb', read_only=True)
con.execute("PRAGMA memory_limit='1024MB'"); con.execute('PRAGMA threads=4')
con.execute("PRAGMA temp_directory='artifacts/checkpoints'")
con.register('query_keys', frame)

SQL = open('src/blocking.py', encoding='utf-8').read()
import re
m = re.search(r'_BLOCK_BATCH_SQL = """(.*?)"""', SQL, re.S)
BATCH_SQL = m.group(1)

print('\n--- EXPLAIN ---')
t0=time.time()
ex = con.execute('EXPLAIN ' + BATCH_SQL, [500]).fetchall()
for row in ex:
    print(row[1][:4000])
print('explain s:', round(time.time()-t0,1))

steps = [
 ('key_set + join rows', "SELECT count(*) FROM (SELECT DISTINCT channel, key FROM query_keys q JOIN blocking_keys k ON k.channel=q.channel AND k.key=q.key)"),
 ('joined keys', "SELECT count(DISTINCT (channel,key)) FROM (SELECT DISTINCT channel, key FROM query_keys q JOIN blocking_keys k ON k.channel=q.channel AND k.key=q.key)"),
]
for name, sql in steps:
    t0=time.time()
    try:
        n = con.execute(sql).fetchone()[0]
        print(f'{name}: {n:,}  ({round(time.time()-t0,1)}s, rss={rss()}MB)')
    except Exception as e:
        print(f'{name}: FAILED {type(e).__name__}: {str(e)[:200]} (rss={rss()}MB)')

# full original query
print('\n--- original batch query ---')
t0=time.time()
try:
    res = con.execute(BATCH_SQL, [500]).fetchall()
    print(f'OK rows={len(res):,} ({round(time.time()-t0,1)}s, rss={rss()}MB)')
except Exception as e:
    print(f'FAILED after {round(time.time()-t0,1)}s rss={rss()}MB: {type(e).__name__}: {str(e)[:300]}')
con.close()
