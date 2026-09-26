import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd, duckdb
from src.blocking import extract_blocking_keys, _BLOCK_MATCH_SQL
from src.training import iter_s1_batches
from src.pipeline import compute_corpus_token_idf

idf = compute_corpus_token_idf('student_resource/dataset/train/train_source1.tsv', sample_size=20_000)
s1 = next(iter(iter_s1_batches('student_resource/dataset/train', s1_chunk_size=20_000,
                               max_s1=20_000, s1_offset=1_000_000)))
rows = []
for rec in s1:
    for co, (ch, val) in enumerate(extract_blocking_keys(rec, idf)):
        rows.append((rec['entity_id'], co, ch, val))
frame = pd.DataFrame(rows, columns=['s1_id','chan_order','channel','key']).drop_duplicates(
    subset=['s1_id','channel','key'], keep='first')
keys = frame['key'].drop_duplicates().tolist()
print(f'query_keys={len(frame)} distinct_keys={len(keys)}', flush=True)
del s1, rows; gc.collect()

con = duckdb.connect('artifacts/checkpoints/bench_blocking.duckdb')
con.execute("PRAGMA memory_limit='1024MB'"); con.execute('PRAGMA threads=4')
con.register('query_keys', frame)

def timeit(sql, params=None, runs=2):
    best = 1e9
    for _ in range(runs):
        t0 = time.time(); res = con.execute(sql, params or []); best = min(best, time.time()-t0)
    return best, res

def explain_grep(sql, params=None):
    try:
        plan = con.execute('EXPLAIN ' + sql, params or []).fetchall()
    except Exception as e:
        return f'explain failed: {type(e).__name__}'
    text = ' '.join(str(r) for tup in plan for r in tup).upper()
    return ','.join(t for t in ('ART_INDEX_JOIN','ART_INDEX','INDEX_SCAN','SEQ_SCAN','HASH_JOIN','PIECEWISE','NESTED_LOOP') if t in text)

best, _ = timeit(_BLOCK_MATCH_SQL)
n_base = con.execute('SELECT count(*) FROM _block_matched').fetchone()[0]
con.execute('CREATE OR REPLACE TEMP TABLE _b0 AS ' + _BLOCK_MATCH_SQL.split('AS',1)[1])
print(f'baseline join : {best:.2f}s rows={n_base:,} plan=[{explain_grep(_BLOCK_MATCH_SQL)}]', flush=True)

SQL_PARAM = '''SELECT DISTINCT k.channel, k.key, k.seq, k.entity_id
FROM blocking_keys k WHERE k.key = ANY(?)'''
best, _ = timeit(SQL_PARAM, [keys])
con.execute('CREATE OR REPLACE TEMP TABLE _b2 AS ' + SQL_PARAM, [keys])
n_p = con.execute('SELECT count(*) FROM _b2').fetchone()[0]
print(f'ANY(list param): {best:.2f}s rows={n_p:,} plan=[{explain_grep(SQL_PARAM, [keys])}]', flush=True)

SQL_IN = '''SELECT DISTINCT k.channel, k.key, k.seq, k.entity_id
FROM blocking_keys k WHERE k.key IN (SELECT key FROM query_keys)'''
best, _ = timeit(SQL_IN)
con.execute('CREATE OR REPLACE TEMP TABLE _b3 AS ' + SQL_IN)
n_i = con.execute('SELECT count(*) FROM _b3').fetchone()[0]
print(f'IN(subquery)   : {best:.2f}s rows={n_i:,} plan=[{explain_grep(SQL_IN)}]', flush=True)

for cand in ('_b2', '_b3'):
    missing = con.execute(f'SELECT count(*) FROM (SELECT * FROM _b0 EXCEPT SELECT * FROM {cand})').fetchone()[0]
    print(f'baseline rows missing from {cand}: {missing}', flush=True)
con.close()
