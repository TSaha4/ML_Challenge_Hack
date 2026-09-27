import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl


class PipelineTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_search(self):
        from src.er.model import assign, macro_f05, tune_threshold
        rng = np.random.default_rng(7)
        for _ in range(8):
            scored = pl.DataFrame({'tid': np.repeat(np.arange(15), 3),
                                  's1': np.tile(np.arange(3), 15),
                                  'p': rng.choice([.1, .501, .502, .9], 45)})
            truth = pl.DataFrame({'tid': np.arange(12), 's1': rng.integers(0, 3, 12)})
            universe = pl.Series([0, 1, 2, 3])
            expected = max(macro_f05(assign(scored, t), truth, universe)['macro_f05']
                           for t in [1.01, .1, .501, .502, .9])
            threshold, _ = tune_threshold(scored, truth, universe)
            self.assertAlmostEqual(macro_f05(assign(scored, threshold), truth, universe)['macro_f05'], expected)

    def test_blocker_handles_empty_index(self):
        from src.er.gpu_blocking import GpuIndex
        s1 = pl.DataFrame({'id': [1], 'cty': ['france'], 'core': [''], 'sq': [''], 'ad': ['']})
        blocker = GpuIndex(s1, device='cpu')
        self.assertEqual(blocker.candidates(s1.with_columns(pl.lit("alpha").alias("core"))).height, 0)

    def test_empty_candidates_have_feature_schema(self):
        from src.er.gpu_features import GpuFeaturizer, FEATURES
        from src.er.gpu_blocking import GpuIndex
        from src.er.features import s1_stats
        from src.er.text import normalize_frame
        raw = pl.DataFrame({'id': [1], 'entity_id': ['S1-1'], 'country': ['France'],
                            'business_name': ['alpha'], 'business_address': ['']})
        s1 = normalize_frame(raw)
        tg = s1.with_columns(pl.lit(2).alias('src'))
        empty = GpuIndex(s1, device='cpu').candidates(s1.head(0))
        features = GpuFeaturizer(s1, s1_stats(empty)).featurize(empty, tg)
        self.assertEqual(features.height, 0)
        self.assertEqual(features.columns, ['tid', 's1', *FEATURES])

    def test_error_report_distinguishes_blocker_and_matcher(self):
        from src.er.diagnostics import entity_errors
        truth = pl.DataFrame({'s1': [100, 100, 104], 'tid': [1, 2, 3]})
        candidates = pl.DataFrame({'s1': [100, 104, 108], 'tid': [1, 3, 4]})
        predictions = pl.DataFrame({'s1': [100, 108], 'tid': [1, 4]})
        errors = entity_errors(predictions, candidates, truth, pl.Series([100, 104, 108, 112]))
        rows = {r['s1']: r for r in errors.to_dicts()}
        self.assertEqual(rows[100]['blocking_misses'], 1)
        self.assertEqual(rows[104]['matcher_misses'], 1)
        self.assertEqual(rows[108]['false_positives'], 1)
        self.assertEqual(rows[112]['f05'], 1.)

    def test_model_metadata_keeps_neural_feature(self):
        from src.er.model import save
        class Booster:
            feature_names = ['x', 'nn_p']
            best_iteration = 3
            def save_model(self, path): Path(path).write_text('{}')
        with tempfile.TemporaryDirectory() as td:
            save(Booster(), {'threshold': .73}, td)
            self.assertEqual(json.loads((Path(td)/'meta.json').read_text())['features'], ['x', 'nn_p'])

    def test_prediction_cache_changes_with_model_and_validator_failure_propagates(self):
        from src.er.steps import predict as P
        from src.er import model as M
        from src.er import gpu_features as F
        from src.er.artifacts import FEATURE_VERSION, write_manifest, fingerprint
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pl.DataFrame({'id': [1], 'entity_id': ['S1-00001']}).write_parquet(root/'test_s1.parquet')
            pl.DataFrame({'id': [20000000002], 'entity_id': ['S2-00002']}).write_parquet(root/'test_tg.parquet')
            pl.DataFrame({'s1':[1]}).write_parquet(root/'test_s1stats.parquet')
            (root/'test_cands').mkdir()
            pl.DataFrame({'s1':[1], 'tid':[20000000002]}).write_parquet(root/'test_cands/part-000.parquet')
            write_manifest(root/'test_cands/manifest.json', fingerprint([root/'test_s1.parquet',root/'test_tg.parquet']), feature_version=FEATURE_VERSION)
            (root/'model').mkdir()
            (root/'model/xgb.json').write_text('model one')
            (root/'model/meta.json').write_text('{}')
            validator = root/'validator.py'
            validator.write_text('import sys\nsys.exit(2)\n')
            class Featurizer:
                def __init__(self, *a, **kw): pass
                def featurize(self,c,t): return c
            args=['--model-dir',str(root/'model'),'--out-dir',str(root/'out'),'--skip-validation']
            with patch.object(P,'ART',root), patch.object(P,'wait_for_ram'), patch.object(P,'lower_priority'), \
                 patch.object(F,'GpuFeaturizer',Featurizer), \
                 patch.object(M,'load',return_value=(SimpleNamespace(feature_names=['x']),{'threshold':.5,'best_iteration':1})), \
                 patch.object(M,'predict',return_value=np.array([.9])) as score:
                P.main(args)
                P.main(args)
                self.assertEqual(score.call_count,1)
                (root/'model/xgb.json').write_text('model two')
                P.main(args)
                self.assertEqual(score.call_count,2)
                with self.assertRaises(SystemExit) as exc:
                    P.main(args[:-1]+['--validator',str(validator)])
                self.assertEqual(exc.exception.code,2)
                manifest = json.loads((root/'out/submission_manifest.json').read_text())
                self.assertEqual(manifest['validator_status'], 'failed')
                self.assertEqual(manifest['threshold'], .5)
                self.assertFalse(manifest['threshold_overridden'])
                write_manifest(root/'test_cands/manifest.json', fingerprint([root/'test_s1.parquet',root/'test_tg.parquet']), feature_version=FEATURE_VERSION, settings={'cap':1000})
                with self.assertRaisesRegex(ValueError, 'blocking settings differ'):
                    P.main(args)
            self.assertEqual((root/'out/matching_results.tsv').read_text(), 'source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00002\n')


if __name__ == '__main__': unittest.main()

class BoundaryTests(unittest.TestCase):
    def test_calibration_empty_candidates_and_all_singletons(self):
        from src.er.model import tune_threshold, assign, macro_f05
        empty = pl.DataFrame(schema={'s1':pl.Int64,'tid':pl.Int64,'p':pl.Float32})
        truth = empty.select('s1','tid')
        for scored in [empty, pl.DataFrame({'s1':[1],'tid':[2],'p':[1.]})]:
            threshold, curve = tune_threshold(scored, truth, pl.Series([1,3]))
            self.assertEqual(assign(scored, threshold).height,0)
            self.assertEqual(curve[0]['macro_f05'],1.)

    def test_content_identity_survives_moving_checkpoint(self):
        from src.er.artifacts import fingerprint
        with tempfile.TemporaryDirectory() as td:
            a,b = Path(td)/'old.pt', Path(td)/'new.pt'
            a.write_bytes(b'model')
            b.write_bytes(a.read_bytes())
            self.assertEqual(fingerprint([a]),fingerprint([b]))

    def test_blocking_manifest_rejects_changed_records(self):
        from src.er.artifacts import require_current_blocking,write_manifest,fingerprint,FEATURE_VERSION
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            inputs=[root/'train_s1.parquet',root/'train_tg.parquet']
            for p in inputs: p.write_bytes(b'old')
            write_manifest(root/'train_cands/manifest.json',fingerprint(inputs),feature_version=FEATURE_VERSION)
            require_current_blocking(root,'train')
            inputs[0].write_bytes(b'new')
            with self.assertRaises(ValueError): require_current_blocking(root,'train')
