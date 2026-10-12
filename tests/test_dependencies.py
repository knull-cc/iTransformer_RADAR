"""Regression tests for startup without optional Reformer dependencies."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OptionalDependencyTests(unittest.TestCase):
    def run_without_reformer(self, source):
        script = textwrap.dedent('''
            import importlib.abc
            import sys

            class MissingReformer(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'reformer_pytorch' or fullname.startswith('reformer_pytorch.'):
                        raise ModuleNotFoundError("No module named 'reformer_pytorch'", name='reformer_pytorch')

            sys.meta_path.insert(0, MissingReformer())
        ''') + '\n' + textwrap.dedent(source)
        return subprocess.run([sys.executable, '-c', script], cwd=PROJECT_ROOT,
                              text=True, capture_output=True)

    def test_itransformer_and_radar_start_without_reformer(self):
        result = self.run_without_reformer('''
            from types import SimpleNamespace
            import torch
            from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
            from model.iTransformer import Model

            args = SimpleNamespace(seq_len=24, pred_len=12, output_attention=False,
                use_norm=True, d_model=16, embed='timeF', freq='h', dropout=0.,
                class_strategy='projection', factor=1, n_heads=2, d_ff=32,
                activation='gelu', e_layers=1, features='M', use_radar=True)
            model = Model(args).eval()
            with torch.no_grad():
                output = model(torch.randn(2, 24, 3), None, None, None)
            assert output.shape == (2, 12, 3)
            assert torch.isfinite(output).all()
            assert 'reformer_pytorch' not in sys.modules
            print('RADAR startup passed without reformer_pytorch')
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('startup passed', result.stdout)

    def test_reformer_reports_installation_instruction_when_selected(self):
        result = self.run_without_reformer('''
            from layers.SelfAttention_Family import ReformerLayer
            try:
                ReformerLayer(None, 16, 2)
            except ModuleNotFoundError as error:
                assert 'pip install reformer-pytorch' in str(error), str(error)
                print('Optional dependency instruction passed')
            else:
                raise AssertionError('Selecting Reformer must require its optional dependency')
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('instruction passed', result.stdout)


class ETTh1ScriptTests(unittest.TestCase):
    def run_script(self, fail=False, visible_devices=None):
        with tempfile.TemporaryDirectory(prefix='radar-script-') as directory:
            root = Path(directory)
            capture = root / 'calls.jsonl'
            python = root / 'python'
            python.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''
                import json
                import os
                import sys
                with open(os.environ['RADAR_CAPTURE_PATH'], 'a') as stream:
                    stream.write(json.dumps({'arguments': sys.argv[1:],
                        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\\n')
                if os.environ.get('RADAR_TEST_FAIL') == '1':
                    raise SystemExit(7)
            '''))
            python.chmod(0o755)
            environment = os.environ.copy()
            environment['PATH'] = str(root) + os.pathsep + environment['PATH']
            environment['RADAR_CAPTURE_PATH'] = str(capture)
            environment['RADAR_TEST_FAIL'] = '1' if fail else '0'
            if visible_devices is None:
                environment.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                environment['CUDA_VISIBLE_DEVICES'] = visible_devices
            result = subprocess.run(['bash', str(PROJECT_ROOT / 'scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh'),
                                     '--use_radar', '1'], cwd=PROJECT_ROOT, env=environment,
                                    capture_output=True, text=True)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            return result, calls

    def test_radar_flag_single_gpu_and_original_forecast_settings(self):
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 4)
        for call, horizon, width in zip(calls, ['96', '192', '336', '720'], ['256', '256', '512', '512']):
            args = call['arguments']
            self.assertEqual(call['gpu'], '0')
            self.assertEqual(args[args.index('--use_radar') + 1], '1')
            self.assertEqual(args[args.index('--pred_len') + 1], horizon)
            self.assertEqual(args[args.index('--d_model') + 1], width)
            self.assertEqual(args[args.index('--d_ff') + 1], width)

    def test_explicit_gpu_choice_is_preserved(self):
        result, calls = self.run_script(visible_devices='2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(call['gpu'] == '2' for call in calls))

    def test_first_failure_stops_repeated_runs(self):
        result, calls = self.run_script(fail=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 1)


class PandasCompatibilityTests(unittest.TestCase):
    def test_calendar_encoding_in_all_csv_loaders(self):
        import numpy as np
        import pandas as pd
        from data_provider.data_loader import (
            Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred,
        )

        with tempfile.TemporaryDirectory(prefix='radar-calendar-') as directory:
            points = np.arange(180)
            frame = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=180, freq='h'),
                'feature': np.sin(points / 5), 'OT': np.cos(points / 7),
            })
            frame.to_csv(Path(directory) / 'series.csv', index=False)
            for loader in [Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred]:
                with self.subTest(loader=loader.__name__):
                    dataset = loader(root_path=directory, data_path='series.csv',
                                     flag='pred' if loader is Dataset_Pred else 'train',
                                     size=[12, 6, 6], features='M', target='OT',
                                     timeenc=0, freq='h')
                    x, y, x_mark, y_mark = dataset[0]
                    self.assertEqual(x.shape, (12, 2))
                    self.assertEqual(x_mark.shape[0], 12)
                    self.assertTrue(np.isfinite(x).all())
                    self.assertTrue(np.isfinite(x_mark).all())


if __name__ == '__main__':
    unittest.main()
