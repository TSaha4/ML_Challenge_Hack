import sys, time, gc, threading
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import psutil, pandas as pd, duckdb
from src.blocking import DuckDBBlocker, extract_blocking_keys
from src.training import iter_s1_batches
from src.pipeline import compute_corpus_token_idf

proc = psutil.Process()
peak = {'rss': 0}
stop = False
def sampler():
    while not stop:
        peak['rss'] = max(peak['rss'], proc.memory_info().rss)
        time.sleep(0.05)
th = threading.Thread(target=sampler, daemon=True); th.start()
rss = lambda: round(proc.memory_info().rss/1e6)

idf = compute_corpus_token_idf('student_resource/dataset/train/train_source1.tsv', sample_size=20_000)
s1 = next(iter(iter_s1_batches('student_resource/dataset/train', s1_chunk_size=20_000,
                               max_s1=20_000, s1_offset=1_000_000)))
key_rows = []
for rec in s1:
    for co, (ch, val) in enumerate(extract_blocking_keys(rec, idf)):
        key_rows.append((rec['entity_id'], co, ch, val))
frame = pd.DataFrame(key_rows, columns=['s1_id','chan_order','channel','key'])
frame = frame.drop_duplicates(subset=['s1_id','channel','key'], keep='first')
print(f's1={len(s1)} qkeys={len(frame)} rss={rss()}')
del s1, key_rows; gc.collect()

def run(name, sql, params=None, fetch=False):
    t0 = time.time(); b0 = peak['rss']
    try:
        cur = con.execute(sql, params or [])
        if fetch:
            res = cur.fetchall()
        else:
            res = cur.fetchone()
        dt = time.time()-t0
        print(f'  OK  {name}: {res if not fetch else str(len(res))+" rows"} ({dt:.1f}s, peakRSS={round(peak["rss"]/1e6)}MB)')
        return res
    except Exception as e:
        print(f'  FAIL {name}: {type(e).__name__}: {str(e)[:160]} ({time.time()-t0:.1f}s, rss={rss()}MB)')
        return None

con = duckdb.connect('artifacts/checkpoints/bench_blocking.duckdb', read_only=True)
con.execute("PRAGMA memory_limit='1024MB'"); con.execute('PRAGMA threads=4')
con.execute("PRAGMA temp_directory='artifacts/checkpoints'")
con.register('query_keys', frame)

print('--- sizes ---')
run('distinct qkeys', "SELECT count(*) FROM (SELECT DISTINCT channel, key FROM query_keys)")
run('join output N', "SELECT count(*) FROM (SELECT DISTINCT q.channel, q.key FROM query_keys q JOIN blocking_keys k ON k.channel=q.channel AND k.key=q.key)")

print('--- two-step: materialize matched, then window over it ---')
run('step1 create _matched', "CREATE OR REPLACE TEMP TABLE _matched AS SELECT DISTINCT q.channel, q.key, k.seq FROM query_keys q JOIN blocking_keys k ON k.channel=q.channel AND k.key=q.key")
run('window over _matched', "SELECT count(*) FROM (SELECT row_number() OVER (PARTITION BY channel, key ORDER BY seq) AS rk FROM _matched)")
run('kept rows (rk<=500)', "SELECT count(*) FROM (SELECT row_number() OVER (PARTITION BY channel, key ORDER BY seq) AS rk FROM _matched) WHERE rk <= 500")

print('--- full two-step final query ---')
STEP2 = """
WITH ranked AS (
    SELECT channel, key, seq, row_number() OVER (PARTITION BY channel, key ORDER BY seq) AS rk
    FROM _matched
), kept AS (SELECT * FROM ranked WHERE rk <= ?)
SELECT q.s1_id, k.key, count(DISTINCT k.channel) AS cc, min(k.rk)-1 AS mr,
       min(q.chan_order * 1000000 + k.rk) AS ok, string_agg(DISTINCT k.channel, '|') AS chans
FROM query_keys q
JOIN kept k ON k.channel = q.channel AND k.key = q.key
GROUP BY q.s1_id, k.key
"""
run('step2 group-by rows', STEP2, [500], fetch=True)

con.execute('DROP TABLE IF EXISTS _matched')
con.unregister('query_keys'); con.close()
gc.collect()
print(f'final rss={rss()}MB')
stop = True
