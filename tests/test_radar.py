"""Behavioral tests for the lightweight RADAR v5.1 input filter."""

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
            dict(receptive_field=0), dict(anchor_window=0),
            dict(dependency_decay=0), dict(gamma=0), dict(mean_weight=-1),
            dict(variance_weight=0), dict(risk_normalization_epsilon=0),
            dict(repair_epsilon=0),
        ]
        for setting in [
                'dependency_decay', 'gamma', 'mean_weight', 'variance_weight',
                'risk_normalization_epsilon', 'repair_epsilon']:
            for value in [float('nan'), float('inf'), float('-inf')]:
                invalid.append({setting: value})
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                RADAR(96, **arguments)


class RiskAndRepairTests(unittest.TestCase):
    def test_risk_is_min_max_normalized_per_instance_and_variate(self):
        radar = diagnostic_radar(96)
        risk = torch.tensor([
            [[0.2, 4.0], [0.3, 2.0], [0.8, 8.0], [0.5, 6.0]],
            [[9.0, 0.4], [3.0, 0.1], [6.0, 0.7], [0.0, 1.0]],
        ], dtype=torch.float64)

        normalized, risk_min, risk_max = radar._normalize_risk(risk)
        expected_min = risk.amin(dim=1, keepdim=True)
        expected_max = risk.amax(dim=1, keepdim=True)
        expected = (
            (risk - expected_min)
            / (expected_max - expected_min + radar.risk_normalization_epsilon)
        )

        torch.testing.assert_close(risk_min, expected_min)
        torch.testing.assert_close(risk_max, expected_max)
        torch.testing.assert_close(normalized, expected)
        torch.testing.assert_close(
            normalized.amin(dim=1), torch.zeros_like(normalized[:, 0]))
        self.assertTrue((normalized >= 0).all())
        self.assertTrue((normalized < 1).all())

    def test_equal_chunk_risks_normalize_to_zero(self):
        radar = diagnostic_radar(96)
        risk = torch.full((2, radar.num_chunks, 3), 0.42)

        normalized, risk_min, risk_max = radar._normalize_risk(risk)

        torch.testing.assert_close(normalized, torch.zeros_like(normalized))
        torch.testing.assert_close(risk_min, risk[:, :1])
        torch.testing.assert_close(risk_max, risk[:, :1])

    def test_repair_strength_combines_relative_rank_and_absolute_evidence(self):
        radar = diagnostic_radar(96)
        history = smooth_history(batch=2, channels=3)
        history[:, -8:, 0] += torch.tensor(
            [4., -4., 4., -4., 4., -4., 4., -4.])

        radar(history)
        diagnostics = radar.last_diagnostics

        expected = diagnostics['normalized_risk'] * diagnostics['risk']
        torch.testing.assert_close(diagnostics['repair_strength'], expected)
        self.assertTrue(
            (diagnostics['repair_strength'] <= diagnostics['risk']).all())
        self.assertTrue(
            (diagnostics['repair_strength'] <= diagnostics['normalized_risk']).all())

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

    def test_clean_periodic_signals_receive_only_small_repairs(self):
        radar = diagnostic_radar(96)
        time = torch.arange(96, dtype=torch.float32)
        frequencies = torch.tensor([0.05, 0.1, 0.15, 0.20678, 0.3, 0.45, 0.6])
        phases = torch.linspace(0, 2 * torch.pi, 9)[:-1]
        frequency, phase = torch.meshgrid(frequencies, phases, indexing='ij')
        history = torch.sin(
            frequency.unsqueeze(-1) * time + phase.unsqueeze(-1)
        ).reshape(-1, 96, 1)

        repaired = radar(history)
        normalized_risk = radar.last_diagnostics['normalized_risk']
        repair_strength = radar.last_diagnostics['repair_strength']

        self.assertTrue(torch.isfinite(repaired).all())
        self.assertTrue((normalized_risk >= 0).all())
        self.assertTrue((normalized_risk < 1).all())
        torch.testing.assert_close(
            normalized_risk.amin(dim=1),
            torch.zeros_like(normalized_risk[:, 0]),
        )
        self.assertTrue((normalized_risk.amax(dim=1) > 0).all())
        self.assertLess(repair_strength.max().item(), 0.04)
        self.assertLess((repaired - history).abs().max().item(), 0.03)

    def test_affine_repair_matches_the_distribution_level_formula(self):
        torch.manual_seed(17)
        radar = diagnostic_radar(96)
        history = torch.randn(2, 96, 3, dtype=torch.float64)
        strength = torch.linspace(0.0, 0.9, radar.num_chunks, dtype=torch.float64)
        strength = strength.view(1, -1, 1).expand(2, -1, 3).clone()

        repaired, diagnostics = radar._repair(history, strength)

        expected_global_mean = history.mean(dim=1, keepdim=True)
        expected_global_std = history.std(dim=1, keepdim=True, unbiased=False)
        torch.testing.assert_close(
            diagnostics['global_mean'], expected_global_mean)
        torch.testing.assert_close(
            diagnostics['global_std'], expected_global_std)
        torch.testing.assert_close(
            diagnostics['point_repair_strength'],
            strength[:, radar.block_ids, :],
        )
        for index, (start, end) in enumerate(radar.chunk_bounds):
            chunk = history[:, start:end, :]
            chunk_mean = chunk.mean(dim=1, keepdim=True)
            chunk_std = chunk.std(dim=1, keepdim=True, unbiased=False)
            score = strength[:, index:index + 1, :]
            safe_mean = (
                (1 - score) * chunk_mean + score * expected_global_mean)
            safe_std = (
                (1 - score) * chunk_std + score * expected_global_std)
            standardized = (chunk - chunk_mean) / chunk_std
            expected = standardized * safe_std + safe_mean

            torch.testing.assert_close(
                diagnostics['chunk_mean'][:, index:index + 1, :], chunk_mean)
            torch.testing.assert_close(
                diagnostics['chunk_std'][:, index:index + 1, :], chunk_std)
            torch.testing.assert_close(
                diagnostics['safe_mean'][:, index:index + 1, :], safe_mean)
            torch.testing.assert_close(
                diagnostics['safe_std'][:, index:index + 1, :], safe_std)
            torch.testing.assert_close(
                diagnostics['standardized'][:, start:end, :], standardized)
            torch.testing.assert_close(repaired[:, start:end, :], expected)

    def test_singleton_chunk_repair_is_finite_and_keeps_its_value(self):
        radar = diagnostic_radar(1)
        history = torch.tensor([[[3.5], [-2.0]]]).transpose(0, 1)
        strength = torch.full((2, 1, 1), 0.9)

        repaired, diagnostics = radar._repair(history, strength)

        torch.testing.assert_close(
            diagnostics['chunk_std'], torch.zeros_like(diagnostics['chunk_std']))
        torch.testing.assert_close(
            diagnostics['global_std'], torch.zeros_like(diagnostics['global_std']))
        self.assertTrue(torch.isfinite(repaired).all())
        torch.testing.assert_close(repaired, history, rtol=0, atol=0)

    def test_relatively_risky_chunk_uses_calibrated_fused_moments(self):
        radar = diagnostic_radar(96)
        history = (torch.arange(96, dtype=torch.float32) / 10).view(1, 96, 1)
        history[:, 88:92, 0] += torch.tensor([0., 100., -100., 0.])
        original = history.clone()

        repaired = radar(history)
        diagnostics = radar.last_diagnostics

        block_index = 5
        start, end = radar.chunk_bounds[block_index]
        self.assertEqual((start, end), (88, 92))
        torch.testing.assert_close(
            diagnostics['risk'],
            diagnostics['dependency'] * (1 - diagnostics['reliability']),
        )
        expected_normalized = (
            (diagnostics['risk'] - diagnostics['risk_min'])
            / (diagnostics['risk_max'] - diagnostics['risk_min']
               + radar.risk_normalization_epsilon)
        )
        torch.testing.assert_close(
            diagnostics['normalized_risk'], expected_normalized)
        torch.testing.assert_close(
            diagnostics['repair_strength'],
            diagnostics['normalized_risk'] * diagnostics['risk'],
        )
        score = diagnostics['repair_strength'][:, block_index:block_index + 1, :]
        self.assertGreater(score.item(), 0.6)
        self.assertGreater(diagnostics['normalized_risk'][:, -1].item(), 0.99)
        self.assertGreater(diagnostics['repair_strength'][:, -1].item(), 0.8)
        chunk = original[:, start:end, :]
        chunk_mean = chunk.mean(dim=1, keepdim=True)
        chunk_std = chunk.std(dim=1, keepdim=True, unbiased=False)
        global_mean = original.mean(dim=1, keepdim=True)
        global_std = original.std(dim=1, keepdim=True, unbiased=False)
        safe_mean = (1 - score) * chunk_mean + score * global_mean
        safe_std = (1 - score) * chunk_std + score * global_std
        standardized = (chunk - chunk_mean) / chunk_std
        expected = standardized * safe_std + safe_mean

        torch.testing.assert_close(
            diagnostics['chunk_mean'][:, block_index:block_index + 1], chunk_mean)
        torch.testing.assert_close(
            diagnostics['chunk_std'][:, block_index:block_index + 1], chunk_std)
        torch.testing.assert_close(
            diagnostics['global_mean'], global_mean)
        torch.testing.assert_close(
            diagnostics['global_std'], global_std)
        torch.testing.assert_close(
            diagnostics['safe_mean'][:, block_index:block_index + 1], safe_mean)
        torch.testing.assert_close(
            diagnostics['safe_std'][:, block_index:block_index + 1], safe_std)
        torch.testing.assert_close(repaired[:, start:end], expected)

        safest = diagnostics['normalized_risk'].argmin(dim=1, keepdim=True)
        point_blocks = radar.block_ids.view(1, -1, 1).expand_as(original)
        identity_mask = point_blocks == safest.expand_as(original)
        torch.testing.assert_close(
            repaired[identity_mask],
            original[identity_mask],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(history, original, rtol=0, atol=0)

    def test_affine_repair_has_finite_input_gradients(self):
        torch.manual_seed(23)
        radar = RADAR(96)
        history = torch.randn(2, 96, 3, dtype=torch.float64, requires_grad=True)
        strength = torch.linspace(0.0, 0.9, radar.num_chunks, dtype=torch.float64)
        strength = strength.view(1, -1, 1).expand(2, -1, 3)

        repaired, _ = radar._repair(history, strength)
        weights = torch.linspace(
            0.5, 1.5, repaired.numel(), dtype=repaired.dtype).reshape_as(repaired)
        (repaired * weights).sum().backward()

        self.assertIsNotNone(history.grad)
        self.assertTrue(torch.isfinite(history.grad).all())
        self.assertGreater(history.grad.abs().sum().item(), 0)

    def test_end_to_end_soft_repair_has_finite_input_gradients(self):
        torch.manual_seed(29)
        radar = RADAR(96)
        history = torch.randn(2, 96, 3, dtype=torch.float64)
        history[:, -8:, 0] *= 6
        history.requires_grad_()

        repaired = radar(history)
        weights = torch.linspace(
            0.5, 1.5, repaired.numel(), dtype=repaired.dtype).reshape_as(repaired)
        (repaired * weights).sum().backward()

        self.assertIsNotNone(history.grad)
        self.assertTrue(torch.isfinite(history.grad).all())
        self.assertGreater(history.grad.abs().sum().item(), 0)

    def test_constant_identity_has_finite_input_gradients(self):
        radar = RADAR(96)
        history = torch.full(
            (2, 96, 3), 7.0, dtype=torch.float64, requires_grad=True)

        repaired = radar(history)
        repaired.square().sum().backward()

        torch.testing.assert_close(repaired, history, rtol=0, atol=0)
        self.assertIsNotNone(history.grad)
        self.assertTrue(torch.isfinite(history.grad).all())
        self.assertGreater(history.grad.abs().sum().item(), 0)

    def test_risk_and_repair_are_independent_for_each_channel(self):
        radar = diagnostic_radar(96)
        history = torch.full((2, 96, 3), 7.)
        history[:, -8:, 0] += torch.tensor(
            [10., -10., 10., -10., 10., -10., 10., -10.])
        repaired = radar(history)
        normalized_risk = radar.last_diagnostics['normalized_risk']
        repair_strength = radar.last_diagnostics['repair_strength']

        self.assertTrue((normalized_risk[:, -1, 0] > 0.99).all())
        self.assertTrue((repair_strength[:, -1, 0] > 0.5).all())
        torch.testing.assert_close(
            normalized_risk[:, :, 1:],
            torch.zeros_like(normalized_risk[:, :, 1:]),
        )
        torch.testing.assert_close(
            repair_strength[:, :, 1:],
            torch.zeros_like(repair_strength[:, :, 1:]),
        )
        torch.testing.assert_close(repaired[:, :, 1:], history[:, :, 1:], rtol=0, atol=0)

    def test_constant_and_singleton_inputs_are_noops(self):
        for length in [1, 96]:
            radar = diagnostic_radar(length)
            history = torch.full((2, length, 3), 7.)
            repaired = radar(history)
            torch.testing.assert_close(repaired, history, rtol=0, atol=0)
            torch.testing.assert_close(
                radar.last_diagnostics['normalized_risk'],
                torch.zeros_like(radar.last_diagnostics['normalized_risk']),
            )
            torch.testing.assert_close(
                radar.last_diagnostics['repair_strength'],
                torch.zeros_like(radar.last_diagnostics['repair_strength']),
            )
            self.assertTrue(torch.isfinite(radar.last_diagnostics['risk']).all())

    def test_zero_strength_values_keep_precision_and_diagnostics_are_detached(self):
        radar = diagnostic_radar(96)
        history = smooth_history().double() + 1e-10
        history[:, -8:, 0] += torch.tensor(
            [10., -10., 10., -10., 10., -10., 10., -10.], dtype=torch.double)
        repaired = radar(history)
        diagnostics = radar.last_diagnostics

        self.assertEqual(repaired.dtype, history.dtype)
        zero_strength = diagnostics['point_repair_strength'] == 0
        torch.testing.assert_close(
            repaired[zero_strength],
            history[zero_strength],
            rtol=0,
            atol=0,
        )
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
