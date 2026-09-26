import duckdb, sys
b = duckdb.connect(r'artifacts/checkpoints/bench_blocking.duckdb', read_only=True)
t = duckdb.connect(r'artifacts/checkpoints/bench_targets.duckdb', read_only=True)
g = duckdb.connect(r'artifacts/checkpoints/bench_gt.duckdb', read_only=True)
print('blocking records :', b.execute('select count(distinct entity_id) from blocking_keys').fetchone()[0])
print('blocking key rows:', b.execute('select count(*) from blocking_keys').fetchone()[0])
print('target records   :', t.execute('select count(*) from target_records').fetchone()[0])
print('gt rows          :', g.execute('select count(*) from gt').fetchone()[0])
for c in (b, t, g): c.close()
