import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl


class SimilarityTests(unittest.TestCase):
    def test_multiset_overlap_matches_counter(self):
        from src.er.gpu_features import gram_overlap, to_bytes
        a = ['aaaa', 'abababa', '', 'a', 'école', 'abcc', 'abcd']
        b = ['aabb', 'ababcca', '', '', 'ecole', 'aabc', 'abcd']
        for q in (2, 3):
            actual = gram_overlap(*to_bytes(pl.Series(a), 32), *to_bytes(pl.Series(b), 32), q)[0].cpu().numpy()
            expected = []
            for x, y in zip(a, b):
                x, y = x.encode(), y.encode()
                ca = Counter(x[i:i+q] for i in range(max(0, len(x)-q+1)))
                cb = Counter(y[i:i+q] for i in range(max(0, len(y)-q+1)))
                inter = sum((ca & cb).values())
                expected.append(inter / max(1, sum(ca.values())+sum(cb.values())-inter))
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            reverse = gram_overlap(*to_bytes(pl.Series(b), 32), *to_bytes(pl.Series(a), 32), q)[0].cpu().numpy()
            np.testing.assert_allclose(actual, reverse)

    def test_empty_idf_corpus(self):
        import torch
        from src.er.gpu_features import TokenIdf, DEV
        idf = TokenIdf(pl.Series(['', '']), 16, 4)
        self.assertTrue(torch.isfinite(idf(torch.tensor([[123, -1]], device=DEV))).all())

    def test_missing_neural_weights_fail_before_featurization(self):
        from src.er.gpu_features import GpuFeaturizer
        with self.assertRaises(FileNotFoundError):
            GpuFeaturizer(pl.DataFrame(), pl.DataFrame(), nn_path='/missing/matcher.pt')


class CacheTests(unittest.TestCase):
    def test_identity_changes_with_contents_not_threshold(self):
        from src.er.artifacts import fingerprint
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'model.json'
            p.write_text('old')
            a = fingerprint([p], {'features': 2})
            self.assertEqual(a, fingerprint([p], {'features': 2}))
            p.write_text('new')
            self.assertNotEqual(a, fingerprint([p], {'features': 2}))

    def test_reuse_requires_matching_complete_manifest(self):
        from src.er.artifacts import require_manifest, write_manifest
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'manifest.json'
            with self.assertRaises(ValueError): require_manifest(p, 'a')
            write_manifest(p, 'a')
            require_manifest(p, 'a')
            with self.assertRaises(ValueError): require_manifest(p, 'b')


class SplitTests(unittest.TestCase):
    def test_holdout_true_targets_excluded_even_when_blocking_missed(self):
        from src.er.splits import reserved_targets
        c = pl.DataFrame({'tid': [1, 2, 3], 's1': [100, 104, 101]})
        gt = pl.DataFrame({'tid': [1, 2, 4], 's1': [100, 104, 204]})
        self.assertEqual(set(reserved_targets(c.lazy(), gt)['tid']), {1, 2, 4})


class ModelTests(unittest.TestCase):
    def test_exact_threshold_beats_coarse_grid_and_ties_are_atomic(self):
        from src.er.model import assign, macro_f05, tune_threshold
        scored = pl.DataFrame({'tid': [10, 11, 12, 13], 's1': [100, 200, 300, 300],
                               'p': [.911, .909, .92, .92]})
        truth = pl.DataFrame({'tid': [10, 12, 13], 's1': [100, 300, 300]})
        universe = pl.Series([100, 200, 300, 400])
        threshold, curve = tune_threshold(scored, truth, universe)
        self.assertAlmostEqual(macro_f05(assign(scored, threshold), truth, universe)['macro_f05'], 1.)
        self.assertTrue(any(r['macro_f05'] == 1. for r in curve))

    def test_assignment_ties_deterministic(self):
        from src.er.model import assign
        scored = pl.DataFrame({'tid': [1, 1], 's1': [3, 2], 'p': [.9, .9]})
        self.assertEqual(assign(scored, .5)['s1'].to_list(), [2])
        self.assertEqual(assign(scored.reverse(), .5)['s1'].to_list(), [2])


class ExportTests(unittest.TestCase):
    def test_original_ids_preserved_and_empty_rows_written(self):
        from src.er.steps.predict import write_lists
        order = pl.DataFrame({'id': [1, 2], 'entity_id': ['S1-00001', 'S1-00002']})
        targets = pl.DataFrame({'id': [20000000003], 'entity_id': ['S2-00003']})
        pairs = pl.DataFrame({'s1': [1, 1], 'tid': [20000000003, 20000000003]}).lazy()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'out.tsv'
            write_lists(order, pairs, 'source1_entity_id\tmatched_entity_ids\n', out, target_ids=targets.lazy())
            self.assertEqual(out.read_text(), 'source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00003\nS1-00002\t\n')


if __name__ == '__main__': unittest.main()
