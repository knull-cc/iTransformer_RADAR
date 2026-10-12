import unittest
from types import SimpleNamespace

import torch
from torch import nn

from layers.RADAR import RADAR, MaskedMLPImputer, geometric_chunk_lengths
from model.iTransformer import Model
from experiments.exp_basic import Exp_Basic


def config(**changes):
    values = dict(seq_len=96, pred_len=12, output_attention=False, use_norm=True,
                  d_model=16, embed='timeF', freq='h', dropout=0.25,
                  class_strategy='projection', factor=1, n_heads=2,
                  d_ff=32, activation='gelu', e_layers=1, features='M',
                  use_radar=True)
    values.update(changes)
    return SimpleNamespace(**values)


def history(batch=2, channels=3):
    return torch.sin(torch.arange(96).float().view(1, 96, 1) * 0.2).repeat(batch, 1, channels)


class RecentBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return x[:, -1:, :] * self.weight


class ChunkingTests(unittest.TestCase):
    def test_template_and_coverage_for_short_odd_and_long_windows(self):
        self.assertEqual(geometric_chunk_lengths(96), (2, 2, 4, 4, 8, 16, 28, 32))
        self.assertEqual(geometric_chunk_lengths(192), (4, 4, 8, 8, 16, 32, 56, 64))
        for length in list(range(1, 514)) + [720, 1024, 4096]:
            chunks = geometric_chunk_lengths(length)
            self.assertEqual(sum(chunks), length)
            self.assertGreater(min(chunks), 0)
            self.assertEqual(tuple(sorted(chunks)), chunks)
        radar = RADAR(96)
        self.assertEqual(radar.chunk_bounds,
                         ((0, 32), (32, 60), (60, 76), (76, 84),
                          (84, 88), (88, 92), (92, 94), (94, 96)))
        torch.testing.assert_close(radar.membership.sum(dim=1), torch.ones(96))

    def test_invalid_lengths_fail_explicitly(self):
        for length in [0, -1, 2.5]:
            with self.assertRaises(ValueError):
                geometric_chunk_lengths(length)


class ImputerTests(unittest.TestCase):
    def test_masked_values_cannot_leak_through_statistics_or_network(self):
        torch.manual_seed(5)
        imputer = MaskedMLPImputer(96)
        nn.init.normal_(imputer.residual[-1].weight, std=0.02)
        masked = torch.zeros(2, 96, 3, dtype=torch.bool)
        masked[:, 40:60, :] = True
        masked[:, -4:, :] = True
        first = history()
        second = first.clone()
        second[masked] = torch.randn_like(second[masked]) * 1000
        torch.testing.assert_close(imputer(first, masked), imputer(second, masked), rtol=0, atol=0)

    def test_zero_residual_interpolates_gaps_and_extends_endpoints(self):
        imputer = MaskedMLPImputer(8)
        x = torch.arange(8).float().view(1, 8, 1)
        masked = torch.tensor([True, False, True, True, False, False, True, True]).view(1, 8, 1)
        result = imputer(x, masked)
        expected = torch.tensor([1., 1., 2., 3., 4., 5., 5., 5.]).view(1, 8, 1)
        torch.testing.assert_close(result, expected)

    def test_visibility_flag_distinguishes_real_zero_from_missing(self):
        imputer = MaskedMLPImputer(8)
        x = torch.zeros(1, 8, 1)
        seen = []
        handle = imputer.residual[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        masked = torch.zeros_like(x, dtype=torch.bool)
        masked[:, -2:, :] = True
        imputer(x, masked)
        handle.remove()
        torch.testing.assert_close(seen[0][..., 8:], (~masked).transpose(1, 2).float())


class DetectionTests(unittest.TestCase):
    def test_recent_corruption_is_repaired_without_changing_other_observations(self):
        radar, backbone = RADAR(96).eval(), RecentBackbone().eval()
        x = history()
        x[:, -2:, 0] += 10
        original = x.clone()
        with torch.no_grad():
            repaired, _ = radar(x, backbone, backbone)
        diagnostics = radar.last_diagnostics
        self.assertTrue(diagnostics['selected_blocks'][:, -1, 0].all())
        self.assertFalse(diagnostics['selected_blocks'][:, :, 1:].any())
        self.assertTrue((diagnostics['selected_blocks'].sum(dim=1) <= 3).all())
        self.assertLess(repaired[:, -2:, 0].abs().max().item(), 2)
        torch.testing.assert_close(repaired[~diagnostics['repair_mask']], x[~diagnostics['repair_mask']], rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        self.assertTrue(all(not value.requires_grad for value in diagnostics.values()))

    def test_top_three_joint_risks_and_dependency_gate(self):
        radar = RADAR(96)
        dependency = torch.tensor([.09, .11, .13, .13, .13, .13, .14, .14]).view(1, 8, 1)
        reliability = torch.tensor([1., 1., 0., 0., 0., 0., 0., 0.]).view(1, 8, 1)
        error = torch.arange(1, 9).float().view(1, 8, 1)
        selected, risk = radar._select_blocks(dependency, reliability, error)
        self.assertEqual(selected.nonzero()[:, 1].tolist(), [5, 6, 7])
        torch.testing.assert_close(risk, dependency * error)
        # Huge error alone cannot make a low-dependency block eligible.
        error[:, 0] = 1e6
        reliability[:, 0] = 0
        selected, _ = radar._select_blocks(dependency, reliability, error)
        self.assertFalse(selected[:, 0].any())

    def test_unselected_observations_preserve_original_precision(self):
        radar, backbone = RADAR(96).eval(), RecentBackbone().eval()
        x = history().double() + 1e-10
        x[:, -2:, 0] += 10
        with torch.no_grad():
            repaired, _ = radar(x, backbone, backbone)
        mask = radar.last_diagnostics['repair_mask']
        self.assertEqual(repaired.dtype, x.dtype)
        torch.testing.assert_close(repaired[~mask], x[~mask], rtol=0, atol=0)

    def test_no_candidates_and_no_trusted_context_are_noops(self):
        radar, backbone = RADAR(96), RecentBackbone()
        x = torch.full((2, 96, 3), 7.)
        result, _ = radar(x, backbone, backbone)
        torch.testing.assert_close(result, x, rtol=0, atol=0)
        self.assertFalse(radar.last_diagnostics['repair_mask'].any())
        selected, _ = radar._select_blocks(torch.full((2, 8, 3), 1 / 8),
                                            torch.zeros(2, 8, 3), torch.ones(2, 8, 3))
        self.assertFalse(selected.any())

    def test_zero_dependency_does_not_force_repair(self):
        radar, backbone = RADAR(96), RecentBackbone()
        x = history()
        x[:, -2:] += 100
        result, _ = radar(x, lambda data: torch.zeros(data.size(0), 1, data.size(2)), backbone)
        torch.testing.assert_close(result, x, rtol=0, atol=0)
        self.assertFalse(radar.last_diagnostics['selected_blocks'].any())

    def test_projections_avoid_cancellation_and_preserve_modes_rng_and_gradients(self):
        radar, backbone = RADAR(96), RecentBackbone().train()
        backbone.dropout.eval()  # Deliberately mixed module modes.
        backbone.weight.grad = torch.tensor(4.)
        random_state = torch.get_rng_state().clone()
        predictor = lambda data: torch.cat((data[:, -1:, :1], -data[:, -1:, :1]), dim=1)
        with torch.no_grad():
            dependency = radar._dependency(history(), predictor, backbone)
        self.assertEqual(dependency[0, -1, 0].item(), 1.)
        self.assertTrue(backbone.training)
        self.assertFalse(backbone.dropout.training)
        self.assertEqual(backbone.weight.grad.item(), 4.)
        torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)

    def test_reliability_is_stable_for_constants_and_measurement_units(self):
        radar = RADAR(96)
        clean = history()
        clean[:, -2:, 0] += 3
        reliability, _ = radar._reliability(clean)
        rescaled, _ = radar._reliability(clean * 5 + 100)
        torch.testing.assert_close(reliability, rescaled, rtol=2e-4, atol=1e-5)
        constant_reliability, _ = radar._reliability(torch.ones_like(clean))
        torch.testing.assert_close(constant_reliability, torch.ones_like(constant_reliability))

    def test_singleton_and_nonfinite_inputs(self):
        radar, backbone = RADAR(1), RecentBackbone()
        x = torch.ones(2, 1, 3)
        repaired, loss = radar(x, backbone, backbone, compute_reconstruction_loss=True)
        torch.testing.assert_close(repaired, x)
        self.assertTrue(torch.isfinite(loss))
        for value in [float('nan'), float('inf')]:
            invalid = x.clone()
            invalid[0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'finite observations'):
                radar(invalid, backbone, backbone)


class BackboneIntegrationTests(unittest.TestCase):
    def test_disabled_bypasses_plugin_and_keeps_backbone_parameter_initialization(self):
        torch.manual_seed(11)
        disabled = Model(config(use_radar=False)).eval()
        torch.manual_seed(11)
        enabled = Model(config()).eval()
        self.assertIsNone(disabled.radar)
        for name, value in disabled.state_dict().items():
            self.assertFalse(name.startswith('radar.'))
            torch.testing.assert_close(value, enabled.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            expected = disabled._forecast_backbone(history(), None, None, None)[0]
            result = disabled(history(), None, None, None)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_attention_contract_and_auxiliary_training_return(self):
        model = Model(config(output_attention=True))
        x = history()
        prediction, attentions = model(x, None, None, None)
        self.assertEqual(prediction.shape, (2, 12, 3))
        self.assertEqual(len(attentions), 1)
        prediction, loss = model(x, None, None, None, return_radar_loss=True)
        self.assertEqual(loss.shape, (1,))
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).all())

    def test_training_reaches_imputer_and_backbone_in_both_attention_modes(self):
        for output_attention in [False, True]:
            model = Model(config(output_attention=output_attention))
            experiment = object.__new__(Exp_Basic)
            experiment.args = config(output_attention=output_attention)
            experiment.model = model
            prediction, auxiliary = experiment._forward_training(history(), None, None, None)
            (prediction.square().mean() + auxiliary).backward()
            self.assertTrue(model.training and model.radar.training)
            self.assertIsNotNone(model.projector.weight.grad)
            self.assertGreater(model.radar.imputer.residual[-1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))

    def test_frozen_backbone_still_supports_saliency_and_imputer_training(self):
        model = Model(config())
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('radar.'))
        prediction, auxiliary = model(history(), None, None, None, return_radar_loss=True)
        (prediction.square().mean() + auxiliary.mean()).backward()
        self.assertIsNone(model.projector.weight.grad)
        self.assertGreater(model.radar.imputer.residual[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.radar.last_diagnostics['dependency'].sum().item(), 0)

    def test_saliency_normalization_gradient_matches_finite_differences(self):
        model = Model(config()).double().eval()
        x = history(batch=1, channels=1).double().requires_grad_(True)
        output = model._forecast_backbone(x, None, None, None, detach_mean=False)[0].sum()
        gradient = torch.autograd.grad(output, x)[0]
        step = 1e-6
        for position in [0, 45, 95]:
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[:, position, :] += step
            minus[:, position, :] -= step
            with torch.no_grad():
                finite_difference = (model._forecast_backbone(plus, None, None, None)[0].sum()
                                     - model._forecast_backbone(minus, None, None, None)[0].sum()) / (2 * step)
            torch.testing.assert_close(gradient[0, position, 0], finite_difference, rtol=1e-5, atol=1e-7)

    def test_evaluation_contexts_and_time_covariates(self):
        model = Model(config()).eval()
        for context in [torch.no_grad, torch.inference_mode]:
            with context():
                x = history()
                marks = torch.randn(2, 96, 4)
                result = model(x, marks, None, None)
                again = model(x, marks, None, None)
            self.assertEqual(result.shape, (2, 12, 3))
            self.assertFalse(result.requires_grad)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result, again, rtol=0, atol=0)

    def test_channel_generalization_and_ms_target_without_future_leakage(self):
        model = Model(config(features='MS')).eval()
        with torch.no_grad():
            for channels in [1, 3, 7]:
                x = history(channels=channels)
                first = model(x, None, torch.zeros(2, 24, channels), None)
                second = model(x, None, torch.full((2, 24, channels), 1000.), None)
                self.assertEqual(first.shape, (2, 12, channels))
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_cpu_mixed_precision_backward_is_finite(self):
        model = Model(config())
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            prediction, auxiliary = model(history(), None, None, None, return_radar_loss=True)
            loss = prediction.square().mean() + auxiliary.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_checkpoint_roundtrip_and_loading_legacy_backbone(self):
        model = Model(config()).eval()
        restored = Model(config()).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(model(history(), None, None, None),
                                       restored(history(), None, None, None), rtol=0, atol=0)
        legacy = Model(config(use_radar=False))
        missing, unexpected = restored.load_state_dict(legacy.state_dict(), strict=False)
        self.assertTrue(all(name.startswith('radar.') for name in missing))
        self.assertFalse(unexpected)


if __name__ == '__main__':
    unittest.main()
