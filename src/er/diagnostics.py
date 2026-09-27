"""Per-entity errors separating retrieval misses from matcher errors."""
import polars as pl


def entity_errors(predictions, candidates, truth, universe):
    u = pl.DataFrame({'s1': universe})
    p = predictions.select('s1', 'tid').unique().join(u, on='s1', how='semi')
    t = truth.select('s1', 'tid').unique().join(u, on='s1', how='semi')
    c = candidates.select('s1', 'tid').unique()
    tp = p.join(t, on=['s1', 'tid'], how='semi').group_by('s1').len('tp')
    hits = t.join(c, on=['s1', 'tid'], how='semi').group_by('s1').len('candidate_hits')
    out = (u.join(t.group_by('s1').len('true_count'), on='s1', how='left')
           .join(p.group_by('s1').len('predicted_count'), on='s1', how='left')
           .join(tp, on='s1', how='left').join(hits, on='s1', how='left').fill_null(0))
    return (out.with_columns(
        (pl.col('predicted_count') - pl.col('tp')).alias('false_positives'),
        (pl.col('true_count') - pl.col('candidate_hits')).alias('blocking_misses'),
        (pl.col('candidate_hits') - pl.col('tp')).alias('matcher_misses'),
        pl.when((pl.col('true_count') == 0) & (pl.col('predicted_count') == 0)).then(1.)
        .otherwise(1.25 * pl.col('tp') / (.25 * pl.col('true_count') + pl.col('predicted_count')).clip(1e-12))
        .alias('f05')).sort(['f05', 's1']))
