"""CPU-light checks for the held-out evaluation launcher; no model execution."""
import copy
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import evaluate_holdout as evaluation
from collect_holdout import import_payload


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.campaign = self.source / 'runs/round2-123'
        self.output = self.root / 'evaluation'
        self.data = self.root / 'data.h5'
        self.data.write_bytes(b'synthetic dataset bytes; never used for model evaluation')
        self.hash = evaluation.digest(self.data)
        self.splits = {'test': 'test-hash', 'test_extrap': 'extrap-hash', 'val': 'val-hash'}
        self.spec = {'dataset_sha256': self.hash, 'draw_studies': [
            {'name': 'R2-A-original', 'checkpoint': 'best-direct.pt', 'steps': 20},
            {'name': 'R2-A-paired', 'checkpoint': 'best-diffusion.pt', 'steps': 50}]}
        evaluation.create(self.source / 'experiments-round-2/campaign.json', self.spec)
        for name in evaluation.TASKS:
            run = self.campaign / name
            filename = 'best-direct.pt' if name == 'R2-A-original' else 'best-diffusion.pt'
            checkpoint = run / filename
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(name.encode())
            cfg = dict(smoke_test_only=False, split='val', checkpoint=str(checkpoint),
                       checkpoint_sha256=evaluation.digest(checkpoint), steps=20 if name == 'R2-A-original' else 50,
                       row_indices=list(range(128)), checkpoint_epoch=200, backbone='original' if name == 'R2-A-original' else 'paired',
                       data_sha256=self.hash, split_hashes=self.splits)
            evaluation.create(run / 'config.json', cfg)
            evaluation.create(run / 'complete.json', {'status': 'complete', 'smoke_test_only': False})
        evaluation.create(self.campaign / 'selection.json', {'checkpoint_sha256': 'reference', 'steps': 20})
        for split in evaluation.SPLITS:
            root = self.campaign / ('final-' + split)
            evaluation.create(root / 'complete.json', {'status': 'complete'})
            evaluation.create(root / 'metrics.json', {'checkpoint_sha256': 'reference', 'sizes': {
                str(k): {'ensemble': {'designs': evaluation.EXPECTED_COUNTS[split]}} for k in (1, 3)}})
        self.configuration = dict(source_root=str(self.source), source_campaign=str(self.campaign),
                                  output_root=str(self.output), data=str(self.data),
                                  source_sha256={'experiments-round-2/campaign.json': evaluation.digest(self.source / 'experiments-round-2/campaign.json')})

    def tearDown(self):
        self.temp.cleanup()

    def choose(self, runs, sizes):
        self.assertEqual(len(runs), 1)
        self.assertIn(runs[0].name, evaluation.TASKS[2:])
        self.assertEqual(sizes, [1, 3])
        cfg = evaluation.read(runs[0] / 'config.json')
        return dict(checkpoint=cfg['checkpoint'], checkpoint_sha256=cfg['checkpoint_sha256'],
                    steps=20, report_sizes=sizes, seed_start=6180, run=str(runs[0].resolve()),
                    selection_split='val', data_sha256=self.hash, split_hashes=self.splits, smoke_test_only=False)

    def test_prepare_keeps_all_trials_and_original_files(self):
        originals = {str(p): evaluation.digest(p) for p in self.source.rglob('*') if p.is_file()}
        plan = evaluation.prepare(self.configuration, self.choose)
        self.assertEqual([c['name'] for c in plan['cases']], list(evaluation.TASKS))
        self.assertEqual(plan['cases'][0]['selection']['report_sizes'], [1, 2, 3, 4, 8, 16])
        self.assertFalse(plan['cases'][0]['selection']['full_validation_confirmation'])
        self.assertEqual(originals, {str(p): evaluation.digest(p) for p in self.source.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            evaluation.prepare(self.configuration, self.choose)

    def test_historical_steps_cannot_change(self):
        run = self.campaign / 'R2-A-original'
        cfg = evaluation.read(run / 'config.json')
        cfg['steps'] = 50
        (run / 'config.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            evaluation.historical_selection(run, self.spec, [1, 3])

    def test_historical_checkpoint_change_is_rejected(self):
        run = self.campaign / 'R2-A-original'
        (run / 'best-direct.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            evaluation.historical_selection(run, self.spec, [1, 3])

    def test_commands_use_both_full_splits_and_frozen_settings(self):
        plan = evaluation.prepare(self.configuration, self.choose)
        for index in range(len(plan['cases'])):
            commands = list(evaluation.commands_for(plan, index))
            self.assertEqual([c[0] for c in commands], list(evaluation.SPLITS))
            for split, target, command in commands:
                self.assertIn('--confirm-test', command)
                self.assertEqual(command[command.index('--limit') + 1], '0')
                self.assertEqual(command[command.index('--seed-start') + 1], '6180')
                self.assertEqual(command[command.index('--steps') + 1], str(plan['cases'][index]['selection']['steps']))
                self.assertIn(split, str(target))

    def test_changed_frozen_manifest_is_rejected_before_execution(self):
        plan = evaluation.prepare(self.configuration, self.choose)
        Path(plan['cases'][0]['selection_manifest']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed after freezing'):
            evaluation.worker(self.output, 0)

    def test_dataset_mismatch_writes_no_plan(self):
        self.data.write_bytes(b'changed dataset')
        with self.assertRaisesRegex(ValueError, 'Dataset checksum'):
            evaluation.prepare(self.configuration, self.choose)
        self.assertFalse((self.output / 'plan.json').exists())

    def summary_payload(self):
        files = {}
        def add(path, value):
            raw = json.dumps(value).encode()
            files[path] = dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        add('evaluation-campaign-complete.json', {'status': 'complete', 'tasks': list(evaluation.TASKS)})
        for name in evaluation.TASKS:
            for split in evaluation.SPLITS:
                prefix = f'evaluations/{name}/{split}/'
                add(prefix + 'complete.json', {'status': 'complete'})
                add(prefix + 'metrics.json', {'sizes': {str(k): {'ensemble': {'designs': evaluation.EXPECTED_COUNTS[split]}} for k in (1, 3)}})
        return dict(files=files, scope='Synthetic metadata fixture', remote_root='/synthetic', archive={'path': '/synthetic.zip'})

    def test_compact_import_is_atomic_and_checks_sums(self):
        payload = self.summary_payload()
        target = self.root / 'local-summary'
        first = next(iter(payload['files'].values()))
        valid_hash = first['sha256']
        first['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            import_payload(payload, target)
        self.assertFalse(target.exists())
        first['sha256'] = valid_hash
        import_payload(payload, target)
        self.assertTrue((target / 'import-scope.json').exists())
        with self.assertRaises(FileExistsError):
            import_payload(payload, target)


if __name__ == '__main__':
    unittest.main()
