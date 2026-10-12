"""Batched evaluation must retain every sample and its metric weight."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from data_provider.data_factory import data_provider, data_dict
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from model.iTransformer import Model


class WindowFixture(Dataset):
    scale = True

    def __init__(self, **kwargs):
        self.seq_len, self.label_len, self.pred_len = kwargs['size']

    def __len__(self):
        return 11

    def __getitem__(self, index):
        x = torch.tensor([index + 1., index + 2.]).repeat(self.seq_len, 1)
        y = torch.ones(self.label_len + self.pred_len, 2)
        x_mark = torch.full((self.seq_len, 4), index + 1.)
        y_mark = torch.full((self.label_len + self.pred_len, 4), index + 1.)
        return x, y, x_mark, y_mark

    def inverse_transform(self, values):
        if values.ndim != 2:
            raise ValueError('The scaler accepts a two-dimensional array')
        return values * 3 + 5


class LastObservation(nn.Module):
    def forward(self, x, x_mark, x_dec, y_mark):
        return x[:, -1:, :].repeat(1, 2, 1)


class MarkerWindowFixture(WindowFixture):
    def __getitem__(self, index):
        x, _, x_mark, y_mark = super().__getitem__(index)
        y = torch.full((self.label_len + self.pred_len, 2), index + 1.)
        return x, y, x_mark, y_mark


class MarkerForecast(nn.Module):
    def forward(self, x, x_mark, x_dec, y_mark):
        return x_mark[:, :1, :1].repeat(1, 2, 1)


def arguments(**changes):
    values = dict(data='custom', embed='timeF', freq='h', root_path='unused',
                  data_path='unused.csv', seq_len=6, label_len=3, pred_len=2,
                  features='M', target='OT', batch_size=4, num_workers=0,
                  use_amp=False, output_attention=False, inverse=False,
                  channel_independence=False, partial_start_index=0, enc_in=2)
    values.update(changes)
    return SimpleNamespace(**values)


class EvaluationBatchTests(unittest.TestCase):
    def test_validation_and_test_keep_the_final_partial_batch(self):
        with patch.dict(data_dict, {'custom': WindowFixture}), redirect_stdout(io.StringIO()):
            for flag in ['val', 'test']:
                with self.subTest(flag=flag):
                    dataset, loader = data_provider(arguments(), flag)
                    batches = list(loader)
                    self.assertEqual(loader.batch_size, 4)
                    self.assertFalse(loader.drop_last)
                    self.assertEqual([batch[0].size(0) for batch in batches], [4, 4, 3])
                    actual = torch.cat([batch[0][:, 0, 0] for batch in batches])
                    torch.testing.assert_close(actual, torch.arange(1., 12.))

    def test_training_still_uses_requested_batch_and_prediction_stays_single_window(self):
        with patch.dict(data_dict, {'custom': WindowFixture}), \
                patch('data_provider.data_factory.Dataset_Pred', WindowFixture), \
                redirect_stdout(io.StringIO()):
            _, training = data_provider(arguments(), 'train')
            self.assertEqual(training.batch_size, 4)
            self.assertTrue(training.drop_last)
            _, prediction = data_provider(arguments(), 'pred')
            self.assertEqual(prediction.batch_size, 1)
            self.assertFalse(prediction.drop_last)

    def test_validation_loss_weights_samples_in_uneven_batches(self):
        for experiment_type in [Exp_Long_Term_Forecast, Exp_Long_Term_Forecast_Partial]:
            with self.subTest(experiment=experiment_type.__name__), \
                    patch.dict(data_dict, {'custom': WindowFixture}), redirect_stdout(io.StringIO()):
                experiment = object.__new__(experiment_type)
                experiment.args = arguments()
                experiment.device = torch.device('cpu')
                experiment.model = LastObservation()
                dataset, loader = experiment._get_data('val')
                # Average of i^2 + i + 0.5 for i=0..10, over all 11 windows.
                loss = experiment.vali(dataset, loader, nn.MSELoss())
                self.assertAlmostEqual(loss, 40.5, places=5)

    def test_saved_predictions_include_every_sample_with_inverse_transform(self):
        original_cwd = Path.cwd()
        for experiment_type, module in [
            (Exp_Long_Term_Forecast, 'experiments.exp_long_term_forecasting'),
            (Exp_Long_Term_Forecast_Partial, 'experiments.exp_long_term_forecasting_partial'),
        ]:
            for inverse in [False, True]:
                with self.subTest(experiment=experiment_type.__name__, inverse=inverse), \
                        tempfile.TemporaryDirectory(prefix='radar-batches-') as directory, \
                        patch.dict(data_dict, {'custom': WindowFixture}), \
                        patch(module + '.visual'), redirect_stdout(io.StringIO()):
                    experiment = object.__new__(experiment_type)
                    experiment.args = arguments(inverse=inverse)
                    experiment.device = torch.device('cpu')
                    experiment.model = LastObservation()
                    try:
                        os.chdir(directory)
                        experiment.test('batched')
                        predictions = np.load('results/batched/pred.npy')
                        targets = np.load('results/batched/true.npy')
                        metrics = np.load('results/batched/metrics.npy')
                    finally:
                        os.chdir(original_cwd)
                    expected = np.array([[i + 1., i + 2.] for i in range(11)], dtype=np.float32)
                    expected = np.repeat(expected[:, None, :], 2, axis=1)
                    expected_target = np.ones_like(expected)
                    if inverse:
                        expected = expected * 3 + 5
                        expected_target = expected_target * 3 + 5
                    self.assertEqual(predictions.shape, (11, 2, 2))
                    np.testing.assert_array_equal(predictions, expected)
                    np.testing.assert_array_equal(targets, expected_target)
                    self.assertAlmostEqual(metrics[1], 40.5 * (9 if inverse else 1), places=5)

    def test_radar_predictions_do_not_depend_on_evaluation_batch_grouping(self):
        torch.manual_seed(21)
        args = SimpleNamespace(seq_len=12, pred_len=2, output_attention=False,
            use_norm=True, d_model=16, embed='timeF', freq='h', dropout=0.,
            class_strategy='projection', factor=1, n_heads=2, d_ff=32,
            activation='gelu', e_layers=1, features='M', use_radar=True)
        model = Model(args).eval()
        x, marks = torch.randn(11, 12, 2), torch.randn(11, 12, 4)
        with torch.no_grad():
            single = torch.cat([model(x[i:i+1], marks[i:i+1], None, None) for i in range(11)])
            batched = torch.cat([model(x[i:i+4], marks[i:i+4], None, None) for i in range(0, 11, 4)])
        torch.testing.assert_close(single, batched, rtol=1e-5, atol=1e-5)

    def test_partial_channel_independence_keeps_each_samples_time_markers(self):
        original_cwd = Path.cwd()
        module = 'experiments.exp_long_term_forecasting_partial'
        with tempfile.TemporaryDirectory(prefix='radar-partial-markers-') as directory, \
                patch.dict(data_dict, {'custom': MarkerWindowFixture}), \
                patch(module + '.visual'), redirect_stdout(io.StringIO()):
            experiment = object.__new__(Exp_Long_Term_Forecast_Partial)
            experiment.args = arguments(channel_independence=True)
            experiment.device = torch.device('cpu')
            experiment.model = MarkerForecast()
            dataset, loader = experiment._get_data('val')
            self.assertEqual(experiment.vali(dataset, loader, nn.MSELoss()), 0.)
            try:
                os.chdir(directory)
                experiment.test('partial-markers')
                predictions = np.load('results/partial-markers/pred.npy')
                targets = np.load('results/partial-markers/true.npy')
            finally:
                os.chdir(original_cwd)
        expected = np.arange(1., 12., dtype=np.float32)[:, None, None]
        expected = np.broadcast_to(expected, (11, 2, 2))
        np.testing.assert_array_equal(predictions, expected)
        np.testing.assert_array_equal(targets, expected)


if __name__ == '__main__':
    unittest.main()
