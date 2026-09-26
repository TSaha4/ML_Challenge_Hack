import duckdb, os, time, json
p = r'artifacts/checkpoints/bench_blocking.duckdb'
print('db size MB:', round(os.path.getsize(p)/1e6,1))
con = duckdb.connect(p, read_only=True)
con.execute("PRAGMA memory_limit='1024MB'"); con.execute('PRAGMA threads=4')
t0=time.time()
print('rows / distinct keys:', con.execute("SELECT count(*), count(DISTINCT (channel, key)) FROM blocking_keys").fetchone())
print('per-channel rows    :', con.execute('SELECT channel, count(*) FROM blocking_keys GROUP BY 1 ORDER BY 2 DESC').fetchall())
print('block stats (max/avg/p99/sum/n):', con.execute('''
    SELECT max(c), CAST(avg(c) AS INT), CAST(quantile_cont(c,0.99) AS INT), sum(c), count(*)
    FROM (SELECT count(*) c FROM blocking_keys GROUP BY channel, key)''').fetchone())
print('top blocks          :', con.execute("SELECT channel, substr(key,1,40), count(*) c FROM blocking_keys GROUP BY 1,2 ORDER BY c DESC LIMIT 8").fetchall())
print('blocks > 500        :', con.execute('SELECT count(*) FROM (SELECT count(*) c FROM blocking_keys GROUP BY channel, key HAVING c > 500)').fetchone())
print('elapsed s:', round(time.time()-t0,1))
con.close()
r = json.load(open(r'artifacts/checkpoints/bench_report.json'))
print('report keys:', list(r.keys()))
for k in ('counts','equivalence','ram'):
    if k in r: print(k, '->', json.dumps(r[k])[:600])
