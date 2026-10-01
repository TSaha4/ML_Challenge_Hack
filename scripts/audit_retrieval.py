"""Measure retrieval tradeoffs on the saved CPU screening fixture.

This is a candidate-recall experiment, not a trained-model or leaderboard score.
Run audit_generalization first to create the normalized fixture.
"""
import argparse
import json
import time
from pathlib import Path

import polars as pl
import torch

from src.er.gpu_blocking import GpuIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default='artifacts/data_audit/screening')
    args = parser.parse_args()
    root = Path(args.fixture)
    s1, targets, truth = [pl.read_parquet(root / f'{name}.parquet')
                          for name in ('s1', 'tg', 'truth')]
    torch.set_num_threads(2)
    baseline = pl.read_parquet(root / 'candidates.parquet')
    baseline_hits = baseline.join(truth, on=['s1', 'tid'], how='semi')
    report = {'scope': 'Sampled candidate universe; not production accuracy. '
                       'More candidates require retraining and false-positive evaluation.',
              'true_pairs': truth.height, 'experiments': []}

    def record(name, candidates, elapsed, settings):
        hits = candidates.join(truth, on=['s1', 'tid'], how='semi')
        recovered = hits.join(baseline_hits, on=['s1', 'tid'], how='anti')
        lost = baseline_hits.join(hits, on=['s1', 'tid'], how='anti')
        row = dict(name=name, settings=settings, seconds=elapsed,
                   candidates=candidates.height, true_pairs_retrieved=hits.height,
                   pair_recall=hits.height / truth.height,
                   recovered_vs_baseline=recovered.height, lost_vs_baseline=lost.height)
        report['experiments'].append(row)
        truth.join(hits, on=['s1', 'tid'], how='anti').write_parquet(root / f'{name}_misses.parquet')
        (root / 'retrieval_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)

    record('baseline', baseline, None, dict(cap=100, k=10, k_wide=80, k_char=6))
    for name, settings in [
        ('higher_cap', dict(cap=500, k=10, k_wide=80, k_char=6)),
        ('wider_rerank', dict(cap=100, k=10, k_wide=200, k_char=6)),
        ('more_candidates', dict(cap=100, k=20, k_wide=80, k_char=12)),
    ]:
        start = time.monotonic()
        idx = GpuIndex(s1, cap=settings['cap'], device='cpu')
        candidates = idx.candidates(targets, **{k: v for k, v in settings.items() if k != 'cap'},
                                    slice_targets=500, max_pairs=1_000_000)
        record(name, candidates, time.monotonic() - start, settings)
        del idx, candidates


if __name__ == '__main__':
    main()
