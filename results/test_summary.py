"""Scientific aggregation and extension checks for the comparison generator."""
import json
import hashlib
import copy
from pathlib import Path
import tempfile
import unittest

from update_summary import Collector, RESULTS, comparison_rows, markdown
from model_architectures import architecture_for


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Collector(RESULTS).collect()

    def test_historical_sampler_and_draw_aggregation(self):
        rows = comparison_rows(self.data)
        direct = [r for r in rows if r['method'] == 'Original direct diffusion']
        self.assertEqual({r['steps'] for r in direct}, {20})
        raw = json.loads((RESULTS / 'remaining/11603443/metrics-direct.json').read_text())
        single = next(r for r in direct if r['report'] == 'single-draw average')
        ensemble = next(r for r in direct if r['report'] == 'ensemble')
        self.assertEqual(single['test_relative_l2'], raw['step_counts']['20']['test']['single_sample_mean']['relative_l2'])
        self.assertEqual(ensemble['test_relative_l2'], raw['step_counts']['20']['test']['ensemble']['relative_l2']['mean'])
        self.assertNotEqual(single['test_relative_l2'], ensemble['test_relative_l2'])

    def test_project_plan_numbers_cover_all_methods_and_scores(self):
        expected = {
            'PCA-32 + GP': '1', 'Plain DeepONet': '2', 'Fourier DeepONet': '3',
            'Original direct diffusion': '4', 'Paired diffusion': '4',
            '4R U-Net': '4R', 'Residual diffusion': '6',
            'R2-A-original': 'R2-A', 'R2-A-paired': 'R2-A',
            'R2-B-paired-long': 'R2-B',
            'R2-C-lr-0.0001': 'R2-C', 'R2-C-lr-0.0002': 'R2-C', 'R2-C-lr-0.0004': 'R2-C',
            'R2-D-min-snr-5': 'R2-D',
        }
        for record in self.data['methods'] + self.data['scores']:
            self.assertEqual(record['plan_id'], expected[record['method']])
            if record['plan_id'] in ('4', '6', 'R2-A', 'R2-B', 'R2-C', 'R2-D'):
                self.assertIn('5', record['related_plan_ids'])
        self.assertEqual([r['plan_id'] for r in comparison_rows(self.data)][:3], ['1', '2', '3'])

    def test_round2_selection_and_cost_history(self):
        if not (RESULTS / 'round2/11638349/selection.json').exists():
            self.skipTest('Imported round-two campaign unavailable')
        scores = [s for s in self.data['scores'] if s['method'] == 'R2-B-paired-long']
        heldout = [s for s in scores if s['split'] == 'test']
        self.assertEqual({s['steps'] for s in heldout}, {20})
        self.assertEqual({s['checkpoint'] for s in heldout}, {'best-noise'})
        self.assertEqual(len([s for s in scores if s['split'] == 'val' and s['selected']]), 1)
        method = next(m for m in self.data['methods'] if m['method'] == 'R2-B-paired-long')
        self.assertGreater(method['total_train_seconds'], method['new_train_seconds'])
        for score in self.data['scores']:
            if score['method'].startswith('R2-C') and score['split'] in ('test', 'test_extrap'):
                selection = json.loads((RESULTS / score['source']).parent.parent.joinpath('selection.json').read_text())
                self.assertEqual(selection['selection_split'], 'val')
                self.assertEqual(selection['steps'], score['steps'])

    def test_architecture_parameter_counts_match_recorded_models(self):
        expected = {
            'Plain-DeepONet-w128-r64': (6, 67074),
            'Fourier-DeepONet-w128-r64': (6, 99586),
            'UNet-2ch-b36': (73, 5601674),
            'UNet-4ch-b36': (73, 5602322),
        }
        for profile in self.data['architectures']:
            if profile['profile'] in expected:
                count, parameters = expected[profile['profile']]
                self.assertEqual(profile['learned_modules'], count)
                self.assertEqual(profile['calculated_parameters'], parameters)
                self.assertEqual(len([r for r in profile['layers'] if r['index'] is not None]), count)
            if profile['profile'].startswith('UNet-'):
                layers = profile['layers']
                self.assertEqual(len([r for r in layers if r['operation'].startswith('Conv2d')]), 50)
                self.assertEqual(len([r for r in layers if r['operation'] == 'Linear']), 22)
                attention = next(r for r in layers if r['name'] == 'attn.attn')
                self.assertEqual(attention['input_shape'], 'B x 128 x 144')
                self.assertEqual(layers[-1]['output_shape'], 'B x 2 x 64 x 128')

    def test_sampling_profiles_and_result_coverage(self):
        methods = {m['method']: m for m in self.data['methods']}
        self.assertEqual(methods['R2-A-original']['architecture_profile'], 'UNet-2ch-b36')
        self.assertEqual(methods['R2-A-paired']['architecture_profile'], 'UNet-4ch-b36')
        for name in ('R2-A-original', 'R2-A-paired'):
            self.assertTrue(methods[name]['results_available'].startswith('Validation panel (128)'))
        for name, method in methods.items():
            if name.startswith(('R2-C-', 'R2-D-')):
                self.assertTrue(method['results_available'].startswith('Full validation (881)'))
        self.assertIn('Test + extrapolation', methods['R2-B-paired-long']['results_available'])
        self.assertIsNone(methods['PCA-32 + GP']['learned_modules'])

    def test_architecture_rejects_unmatched_parameter_count(self):
        with self.assertRaisesRegex(ValueError, 'disagrees with saved configuration'):
            architecture_for('4', {'base': 36, 'parameters': 123})

    def test_separate_heldout_scores_attach_to_original_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = RESULTS / 'round2/11638349/R2-C-lr-0.0001'
            trial = root / 'round2/123/R2-C-lr-0.0001'
            trial.mkdir(parents=True)
            for name in ('config.json', 'validation.json', 'complete.json'):
                (trial / name).write_bytes((source / name).read_bytes())
            cfg = json.loads((trial / 'config.json').read_text())
            validation = json.loads((trial / 'validation.json').read_text())
            selected_hash = validation['checkpoints']['best-field']['sha256']
            case = root / 'round2/123/heldout/evaluations/R2-C-lr-0.0001'
            case.mkdir(parents=True)
            selection = dict(run=cfg['output'], checkpoint='/remote/best-field.pt', checkpoint_sha256=selected_hash,
                             selection_split='val', steps=20, report_sizes=[1, 3], seed_start=6180)
            (case / 'selection.json').write_text(json.dumps(selection))
            target = case / 'test'
            target.mkdir()
            (target / 'config.json').write_text(json.dumps(dict(steps=20, selection_manifest_sha256=hashlib.sha256((case / 'selection.json').read_bytes()).hexdigest())))
            (target / 'complete.json').write_text(json.dumps({'status': 'complete'}))
            metrics = {k: {'mean': 0.1} for k in ('relative_l2', 'edge_relative_l2', 'gradient_relative_l2', 'masked_mse')}
            metrics['designs'] = 881
            sizes = {str(k): dict(ensemble=metrics, single_draw_mean={key: 0.1 for key in metrics if key != 'designs'},
                                 estimated_serial_ensemble_seconds=0.1*k, batch_one_single_draw_seconds=0.1) for k in (1, 3)}
            (target / 'metrics.json').write_text(json.dumps(dict(split='test', checkpoint_sha256=selected_hash, sizes=sizes)))
            data = Collector(root).collect()
            self.assertEqual(len(data['methods']), 1)
            heldout = [r for r in data['scores'] if r['split'] == 'test']
            self.assertEqual(len(heldout), 3)
            self.assertTrue(all(r['method_key'] == 'round2/123/R2-C-lr-0.0001/validation.json' for r in heldout))
            self.assertEqual(data['methods'][0]['results_available'], 'Full validation (881); Test')

    def test_large_draw_counts_remain_outside_common_comparison(self):
        data = copy.deepcopy(self.data)
        source = next(r for r in data['scores'] if r['split'] == 'test' and r['selected'])
        data['scores'].append({**source, 'draws': 16, 'report': 'ensemble', 'id': 'synthetic-large-draw-record'})
        self.assertFalse(any(r['draws'] == 16 for r in comparison_rows(data)))

    def test_new_run_is_discovered_and_missing_split_stays_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'new_method/999'
            run.mkdir(parents=True)
            aggregate = {name: {'mean': 0.0} for name in ('relative_l2', 'edge_relative_l2', 'gradient_relative_l2', 'masked_mse')}
            aggregate['designs'] = 881
            (run / 'metrics.json').write_text(json.dumps({'smoke_test_only': False, 'test': aggregate}))
            data = Collector(root).collect()
            rows = comparison_rows(data)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['test_relative_l2'], 0.0)
            self.assertIsNone(rows[0]['test_extrap_relative_l2'])
            self.assertIn('n.a.', markdown(data, rows))
            second = root / 'new_method/1000'
            second.mkdir()
            (second / 'metrics.json').write_text(json.dumps({'test': aggregate}))
            self.assertEqual(len(comparison_rows(Collector(root).collect())), 2)

    def test_smoke_and_unknown_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'smoke-metrics.json').write_text(json.dumps({'smoke_test_only': True, 'test': {}}))
            (root / 'unknown-metrics.json').write_text(json.dumps({'different_metric': 0.3}))
            data = Collector(root).collect()
            self.assertEqual(data['scores'], [])
            self.assertEqual(len(data['warnings']), 1)

    def test_incomplete_draw_output_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'R2-A-new'
            run.mkdir()
            (run / 'metrics.json').write_text(json.dumps({'sizes': {}, 'split': 'val'}))
            data = Collector(root).collect()
            self.assertEqual(data['scores'], [])
            self.assertIn('incomplete', data['warnings'][0])


if __name__ == '__main__':
    unittest.main()
