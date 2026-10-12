"""Behavioral tests for the lightweight RADAR v2 input filter."""

import unittest
from types import SimpleNamespace

import torch

from experiments.exp_basic import Exp_Basic
from layers.RADAR import RADAR, geometric_chunk_lengths
from model.iTransformer import Model


def config(**changes):
    values = dict(seq_len=96, pred_len=12, output_attention=False, use_norm=True,
                  d_model=16, embed='timeF', freq='h', dropout=0.25,
                  class_strategy='projection', factor=1, n_heads=2,
                  d_ff=32, activation='gelu', e_layers=1, features='M',
                  use_radar=True)
    values.update(changes)
    return SimpleNamespace(**values)


def smooth_history(batch=2, channels=3):
    time = torch.arange(96, dtype=torch.float32)
    signal = 0.02 * time + torch.sin(0.15 * time)
    return signal.view(1, 96, 1).repeat(batch, 1, channels)


def diagnostic_radar(*args, **kwargs):
    """Build RADAR with the opt-in detailed diagnostics used by these tests."""
    return RADAR(*args, store_diagnostics=True, **kwargs)


class ChunkingTests(unittest.TestCase):
    def test_template_and_coverage_for_short_odd_and_long_windows(self):
        self.assertEqual(geometric_chunk_lengths(96), (2, 2, 4, 4, 8, 16, 28, 32))
        self.assertEqual(geometric_chunk_lengths(192), (4, 4, 8, 8, 16, 32, 56, 64))
        for length in list(range(1, 514)) + [720, 1024, 4096]:
            chunks = geometric_chunk_lengths(length)
            self.assertEqual(sum(chunks), length)
            self.assertGreater(min(chunks), 0)
            self.assertEqual(tuple(sorted(chunks)), chunks)
        radar = diagnostic_radar(96)
        self.assertEqual(
            radar.chunk_bounds,
            ((0, 32), (32, 60), (60, 76), (76, 84),
             (84, 88), (88, 92), (92, 94), (94, 96)),
        )
        torch.testing.assert_close(radar.membership.sum(dim=1), torch.ones(96))

    def test_short_chunks_expand_to_causal_eight_step_receptive_fields(self):
        radar = diagnostic_radar(96)
        self.assertEqual(
            radar.statistic_bounds,
            ((0, 32), (32, 60), (60, 76), (76, 84),
             (80, 88), (84, 92), (86, 94), (88, 96)),
        )
        self.assertEqual(
            radar.anchor_bounds,
            ((0, 0), (0, 32), (28, 60), (44, 76),
             (48, 80), (52, 84), (54, 86), (56, 88)),
        )

    def test_invalid_configuration_fails_explicitly(self):
        for length in [0, -1, 2.5]:
            with self.assertRaises(ValueError):
                geometric_chunk_lengths(length)
        invalid = [
            dict(top_k=0), dict(receptive_field=0), dict(anchor_window=0),
            dict(dependency_decay=0), dict(gamma=0), dict(mean_weight=-1),
            dict(variance_weight=0), dict(repair_strength=0),
        ]
        for setting in [
                'dependency_decay', 'gamma', 'mean_weight',
                'variance_weight', 'repair_strength']:
            for value in [float('nan'), float('inf'), float('-inf')]:
                invalid.append({setting: value})
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                RADAR(96, **arguments)


class RiskAndRepairTests(unittest.TestCase):
    def test_recent_short_chunk_statistics_use_the_expanded_receptive_field(self):
        radar = diagnostic_radar(96)
        history = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
        repaired = radar(history)
        diagnostics = radar.last_diagnostics

        self.assertEqual(repaired.shape, history.shape)
        torch.testing.assert_close(
            diagnostics['local_mean'][0, -1, 0], history[0, 88:96, 0].mean())
        torch.testing.assert_close(
            diagnostics['local_std'][0, -1, 0],
            history[0, 88:96, 0].std(unbiased=False))

    def test_dependency_is_a_data_independent_inverse_exponential_recency_prior(self):
        radar = diagnostic_radar(96, dependency_decay=3.0)
        first = torch.randn(2, 96, 3)
        second = torch.randn(2, 96, 3) * 100 + 500

        radar(first)
        dependency = radar.last_diagnostics['dependency'].clone()
        radar(second)
        other_dependency = radar.last_diagnostics['dependency']

        gaps = torch.tensor([96 - end for _, end in radar.chunk_bounds])
        expected = torch.exp(-3.0 * gaps / 96).view(1, 8, 1).expand_as(dependency)
        torch.testing.assert_close(dependency, expected)
        torch.testing.assert_close(other_dependency, expected)
        self.assertTrue((dependency[:, 1:] > dependency[:, :-1]).all())
        torch.testing.assert_close(dependency[:, -1], torch.ones_like(dependency[:, -1]))

    def test_latest_reliability_does_not_use_distant_global_history(self):
        radar = diagnostic_radar(96)
        history = smooth_history(batch=1, channels=1)
        radar(history)
        expected = radar.last_diagnostics['reliability'][:, -1].clone()

        changed_distant_history = history.clone()
        changed_distant_history[:, :56] += 1000
        radar(changed_distant_history)
        actual = radar.last_diagnostics['reliability'][:, -1]

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reliability_is_stable_under_affine_unit_changes(self):
        radar = diagnostic_radar(96)
        history = smooth_history(batch=1, channels=1)
        history[:, -8:] += torch.tensor(
            [2., -2., 2., -2., 2., -2., 2., -2.]).view(1, 8, 1)
        radar(history)
        expected = radar.last_diagnostics['reliability'].clone()
        radar(history * 5 + 100)
        actual = radar.last_diagnostics['reliability']
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=2e-4)

        radar(history * 1e-3)
        downscaled = radar.last_diagnostics['reliability']
        torch.testing.assert_close(downscaled, expected, rtol=5e-4, atol=2e-4)

    def test_local_anchor_does_not_materially_flatten_a_smooth_linear_trend(self):
        radar = diagnostic_radar(96)
        history = (100 * torch.arange(96, dtype=torch.float32)).view(1, 96, 1)
        repaired = radar(history)

        self.assertLess((repaired - history).abs().max().item(), 1e-3)
        for start, end in radar.chunk_bounds:
            torch.testing.assert_close(
                repaired[:, start:end].mean(dim=1),
                history[:, start:end].mean(dim=1),
                rtol=0,
                atol=1e-3,
            )

    def test_clean_periodic_signal_is_not_aggressively_smoothed(self):
        radar = diagnostic_radar(96)
        time = torch.arange(96, dtype=torch.float32)
        frequencies = torch.tensor([0.05, 0.1, 0.15, 0.20678, 0.3, 0.45, 0.6])
        phases = torch.linspace(0, 2 * torch.pi, 9)[:-1]
        frequency, phase = torch.meshgrid(frequencies, phases, indexing='ij')
        history = torch.sin(
            frequency.unsqueeze(-1) * time + phase.unsqueeze(-1)
        ).reshape(-1, 96, 1)

        repaired = radar(history)

        self.assertLess((repaired - history).abs().max().item(), 0.02)

    def test_high_frequency_noise_is_softened_without_replacing_chunk_trend(self):
        radar = diagnostic_radar(96)
        history = (torch.arange(96, dtype=torch.float32) / 10).view(1, 96, 1)
        history[:, -8:, 0] += torch.tensor(
            [10., -10., 10., -10., 10., -10., 10., -10.])
        original = history.clone()

        repaired = radar(history)
        diagnostics = radar.last_diagnostics

        self.assertTrue(diagnostics['selected_blocks'][0, -1, 0])
        self.assertEqual(diagnostics['selected_blocks'].sum().item(), 3)
        torch.testing.assert_close(
            diagnostics['risk'],
            diagnostics['dependency'] * (1 - diagnostics['reliability']),
        )
        expected_alpha = torch.exp(-radar.repair_strength * diagnostics['risk'].square())
        expected_alpha = torch.where(
            diagnostics['selected_blocks'], expected_alpha, torch.ones_like(expected_alpha))
        torch.testing.assert_close(diagnostics['alpha'], expected_alpha)
        start, end = radar.chunk_bounds[-1]
        trend = original[:, start:end].mean(dim=1, keepdim=True)
        alpha = diagnostics['alpha'][:, -1:, :]
        expected = trend + alpha * (original[:, start:end] - trend)
        torch.testing.assert_close(repaired[:, start:end], expected)
        torch.testing.assert_close(repaired[:, start:end].mean(dim=1), trend[:, 0])
        self.assertLess(
            repaired[:, start:end].std(unbiased=False).item(),
            original[:, start:end].std(unbiased=False).item() * 0.1,
        )
        torch.testing.assert_close(
            repaired[~diagnostics['repair_mask']],
            original[~diagnostics['repair_mask']],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(history, original, rtol=0, atol=0)

    def test_risk_and_repair_are_independent_for_each_channel(self):
        radar = diagnostic_radar(96)
        history = torch.full((2, 96, 3), 7.)
        history[:, -8:, 0] += torch.tensor(
            [10., -10., 10., -10., 10., -10., 10., -10.])
        repaired = radar(history)
        selected = radar.last_diagnostics['selected_blocks']

        self.assertTrue(selected[:, -1, 0].all())
        self.assertFalse(selected[:, :, 1:].any())
        torch.testing.assert_close(repaired[:, :, 1:], history[:, :, 1:], rtol=0, atol=0)

    def test_constant_and_singleton_inputs_are_noops(self):
        for length in [1, 96]:
            radar = diagnostic_radar(length)
            history = torch.full((2, length, 3), 7.)
            repaired = radar(history)
            torch.testing.assert_close(repaired, history, rtol=0, atol=0)
            self.assertFalse(radar.last_diagnostics['repair_mask'].any())
            self.assertTrue(torch.isfinite(radar.last_diagnostics['risk']).all())

    def test_unselected_values_keep_original_precision_and_diagnostics_are_detached(self):
        radar = diagnostic_radar(96)
        history = smooth_history().double() + 1e-10
        history[:, -8:, 0] += torch.tensor(
            [10., -10., 10., -10., 10., -10., 10., -10.], dtype=torch.double)
        repaired = radar(history)
        diagnostics = radar.last_diagnostics

        self.assertEqual(repaired.dtype, history.dtype)
        torch.testing.assert_close(
            repaired[~diagnostics['repair_mask']],
            history[~diagnostics['repair_mask']],
            rtol=0,
            atol=0,
        )
        self.assertTrue((diagnostics['selected_blocks'].sum(dim=1) <= 3).all())
        self.assertTrue(all(not value.requires_grad for value in diagnostics.values()))

    def test_invalid_input_shape_fails_explicitly(self):
        radar = diagnostic_radar(96)
        for invalid in [torch.ones(96, 3), torch.ones(2, 95, 3)]:
            with self.assertRaisesRegex(ValueError, r'\[batch, seq_len, channels\]'):
                radar(invalid)

    def test_non_finite_input_is_rejected_before_it_can_spread(self):
        radar = diagnostic_radar(96)
        for bad in [float('nan'), float('inf'), float('-inf')]:
            history = smooth_history(batch=1, channels=1)
            history[0, -1, 0] = bad
            with self.subTest(value=bad), self.assertRaisesRegex(
                    ValueError, 'finite observations'):
                radar(history)

    def test_detailed_diagnostics_are_opt_in(self):
        radar = RADAR(96)
        history = smooth_history(batch=1, channels=1)
        compact_score = radar._score(history, detailed=False)
        self.assertIsInstance(compact_score, torch.Tensor)
        self.assertEqual(compact_score.shape, (1, 8, 1))

        radar(history)
        self.assertIsNone(radar.last_diagnostics)

        radar.set_diagnostics(True)
        radar(smooth_history(batch=1, channels=1))
        self.assertIsNotNone(radar.last_diagnostics)
        self.assertIn('risk_distance', radar.last_diagnostics)

        radar.set_diagnostics(False)
        self.assertIsNone(radar.last_diagnostics)


class BackboneIntegrationTests(unittest.TestCase):
    def test_disabled_bypasses_plugin_and_keeps_backbone_initialization(self):
        torch.manual_seed(11)
        disabled = Model(config(use_radar=False)).eval()
        torch.manual_seed(11)
        enabled = Model(config()).eval()
        self.assertIsNone(disabled.radar)
        self.assertEqual(disabled.state_dict().keys(), enabled.state_dict().keys())
        for name, value in disabled.state_dict().items():
            torch.testing.assert_close(value, enabled.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            expected = disabled._forecast_backbone(smooth_history(), None, None, None)[0]
            actual = disabled(smooth_history(), None, None, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_enabled_model_runs_one_backbone_pass_and_adds_no_parameters(self):
        model = Model(config()).eval()
        backbone_calls = []
        handle = model.enc_embedding.register_forward_hook(
            lambda module, inputs, output: backbone_calls.append(1))
        try:
            with torch.no_grad():
                prediction = model(torch.randn(4, 96, 3), None, None, None)
        finally:
            handle.remove()

        self.assertEqual(prediction.shape, (4, 12, 3))
        self.assertEqual(backbone_calls, [1])
        self.assertEqual(sum(parameter.numel() for parameter in model.radar.parameters()), 0)
        self.assertIsNone(model.radar.last_diagnostics)

    def test_attention_contract_and_training_helper_return_only_forecasts(self):
        for output_attention in [False, True]:
            model = Model(config(output_attention=output_attention))
            experiment = object.__new__(Exp_Basic)
            experiment.args = config(output_attention=output_attention)
            experiment.model = model
            prediction = experiment._forward_training(smooth_history(), None, None, None)
            self.assertEqual(prediction.shape, (2, 12, 3))

        prediction, attentions = Model(config(output_attention=True))(
            smooth_history(), None, None, None)
        self.assertEqual(prediction.shape, (2, 12, 3))
        self.assertEqual(len(attentions), 1)

    def test_training_gradient_reaches_the_backbone(self):
        model = Model(config())
        prediction = model(smooth_history(), None, None, None)
        prediction.square().mean().backward()
        self.assertIsNotNone(model.projector.weight.grad)
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_evaluation_contexts_and_time_covariates_are_supported(self):
        model = Model(config()).eval()
        for context in [torch.no_grad, torch.inference_mode]:
            with context():
                x = smooth_history()
                marks = torch.randn(2, 96, 4)
                result = model(x, marks, None, None)
                again = model(x, marks, None, None)
            self.assertEqual(result.shape, (2, 12, 3))
            self.assertFalse(result.requires_grad)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result, again, rtol=0, atol=0)

    def test_channel_generalization_and_future_decoder_values_do_not_affect_filter(self):
        model = Model(config(features='MS')).eval()
        with torch.no_grad():
            for channels in [1, 3, 7]:
                x = smooth_history(channels=channels)
                first = model(x, None, torch.zeros(2, 24, channels), None)
                second = model(x, None, torch.full((2, 24, channels), 1000.), None)
                self.assertEqual(first.shape, (2, 12, channels))
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_cpu_mixed_precision_backward_is_finite(self):
        model = Model(config())
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            prediction = model(smooth_history(), None, None, None)
            loss = prediction.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_checkpoint_roundtrip_and_legacy_backbone_load_strictly(self):
        model = Model(config()).eval()
        restored = Model(config()).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(
                model(smooth_history(), None, None, None),
                restored(smooth_history(), None, None, None),
                rtol=0,
                atol=0,
            )
        legacy = Model(config(use_radar=False))
        restored.load_state_dict(legacy.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
