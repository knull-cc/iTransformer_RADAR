"""RADAR: dependency-aware risk detection and repair in observation space."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


def geometric_chunk_lengths(seq_len):
    """Return deterministic chunk lengths, ordered from newest to oldest.

    Scale the common recency-refined geometric template to the lookback length.
    Largest-remainder allocation preserves every time point, including short or
    odd windows. No boundaries depend on the data, batch, or channel.
    """
    if not isinstance(seq_len, int) or seq_len < 1:
        raise ValueError("RADAR seq_len must be a positive integer")
    if seq_len < 8:
        return (1,) * seq_len
    template = (2, 2, 4, 4, 8, 16, 28, 32)
    lengths = [max(1, seq_len * length // 96) for length in template]
    remainder = seq_len - sum(lengths)
    if remainder > 0:
        order = sorted(range(8), key=lambda i: (seq_len * template[i] % 96, i), reverse=True)
        for i in order[:remainder]:
            lengths[i] += 1
    while sum(lengths) > seq_len:
        i = max(range(8), key=lambda j: (lengths[j], j))
        lengths[i] -= 1
    return tuple(sorted(lengths))


@contextmanager
def _deterministic_backbone(backbone):
    """Temporarily disable dropout, preserving even mixed train/eval modes."""
    modes = [(module, module.training) for module in backbone.modules()]
    try:
        backbone.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


class MaskedMLPImputer(nn.Module):
    """Channel-shared MLP predicting observations from visible history only."""

    def __init__(self, seq_len, hidden_dim=64):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(2 * seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_len),
        )
        # An untrained plugin starts from interpolation; learning adds a residual.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def _interpolate(values, visible):
        # [B, C, L]; endpoint gaps extend the closest visible historical value.
        length = values.size(-1)
        positions = torch.arange(length, device=values.device).view(1, 1, length)
        left = torch.where(visible, positions, -1).cummax(dim=-1).values
        right = torch.where(visible, positions, length).flip(-1).cummin(dim=-1).values.flip(-1)
        left = torch.where(left < 0, right, left)
        right = torch.where(right >= length, left, right)
        left = left.clamp(0, length - 1)
        right = right.clamp(0, length - 1)
        left_values = values.gather(-1, left)
        right_values = values.gather(-1, right)
        fraction = (positions - left).to(values.dtype) / (right - left).clamp_min(1)
        return left_values + fraction * (right_values - left_values)

    def forward(self, x, masked):
        visible = ~masked
        count = visible.sum(dim=1, keepdim=True).clamp_min(1)
        # Mask before computing statistics: hidden targets cannot leak via normalization.
        observed = torch.where(visible, x, torch.zeros_like(x))
        mean = observed.sum(dim=1, keepdim=True) / count
        centered = torch.where(visible, x - mean, torch.zeros_like(x))
        scale = torch.sqrt(centered.square().sum(dim=1, keepdim=True) / count + 1e-5)
        normalized = centered / scale
        values = normalized.transpose(1, 2)
        observed_mask = visible.transpose(1, 2)
        features = torch.cat((values, observed_mask.to(values.dtype)), dim=-1)
        residual = self.residual(features)
        normalized_repair = self._interpolate(values, observed_mask) + residual
        return normalized_repair.transpose(1, 2) * scale + mean


class RADAR(nn.Module):
    """Repair at most three high-dependency, low-reliability blocks per channel.

    ``predictor`` must return forecasts from the *unmodified* backbone, without
    recursively invoking RADAR. Detection is detached; gradients flow through
    the imputer and the final forecasting pass. No future labels are used here.
    """

    def __init__(self, seq_len, top_k=3, reliability_threshold=0.2,
                 gamma=1.0, hidden_dim=64, saliency_probes=2,
                 reconstruction_weight=0.1):
        super().__init__()
        if top_k < 1 or saliency_probes < 1:
            raise ValueError("RADAR top_k and saliency_probes must be positive")
        if not 0 < reliability_threshold < 1 or gamma <= 0:
            raise ValueError("RADAR requires 0 < reliability_threshold < 1 and gamma > 0")
        self.seq_len = seq_len
        self.chunk_lengths = geometric_chunk_lengths(seq_len)
        self.num_chunks = len(self.chunk_lengths)
        self.top_k = min(top_k, self.num_chunks)
        self.reliability_threshold = reliability_threshold
        self.gamma = gamma
        self.saliency_probes = saliency_probes
        self.reconstruction_weight = reconstruction_weight
        self.imputer = MaskedMLPImputer(seq_len, hidden_dim)

        bounds, block_ids, start = [], [], 0
        for i, length in enumerate(reversed(self.chunk_lengths)):
            bounds.append((start, start + length))
            block_ids.extend([i] * length)
            start += length
        self.chunk_bounds = tuple(bounds)  # chronological order, [start, end)
        block_ids = torch.tensor(block_ids, dtype=torch.long)
        membership = F.one_hot(block_ids, self.num_chunks).float()
        self.register_buffer("block_ids", block_ids, persistent=False)
        self.register_buffer("membership", membership, persistent=False)
        self.register_buffer("block_sizes", membership.sum(dim=0), persistent=False)
        self.last_diagnostics = None

    def _block_mean(self, values):
        totals = torch.einsum("blc,lk->bkc", values, self.membership.to(values.dtype))
        return totals / self.block_sizes.to(values.dtype).view(1, -1, 1)

    def _point_mask(self, blocks):
        return blocks[:, self.block_ids, :]

    def _reliability(self, x):
        # The older half avoids diluting recent corruption into its own anchor.
        anchor = x[:, :max(1, self.seq_len // 2), :]
        anchor_mean = anchor.mean(dim=1, keepdim=True)
        anchor_var = anchor.var(dim=1, keepdim=True, unbiased=False)
        anchor_std = torch.sqrt(anchor_var + 1e-5)
        block_mean = self._block_mean(x)
        block_var = self._block_mean((x - block_mean[:, self.block_ids, :]).square())
        block_std = torch.sqrt(block_var + 1e-5)
        wasserstein = (block_mean - anchor_mean).square() + (block_std - anchor_std).square()
        # Dimensionless distance makes gamma independent of measurement units.
        distance = wasserstein / (anchor_var + 1e-5)
        return torch.exp(-self.gamma * distance), anchor_std

    def _dependency(self, x, predictor, backbone):
        # enable_grad is needed inside the existing no_grad validation/test loops.
        # inference_mode(False) also supports callers using inference_mode().
        with torch.inference_mode(False), torch.enable_grad(), \
                torch.autocast(device_type=x.device.type, enabled=False), \
                _deterministic_backbone(backbone):
            probe_x = x.detach().clone().requires_grad_(True)
            predictions = predictor(probe_x)
            squared_gradients = torch.zeros_like(probe_x)
            if predictions.requires_grad:
                # Fixed Rademacher projections estimate the Jacobian norm without
                # materializing [forecast_length, channels, lookback, channels].
                # Using forecast.sum() alone would cancel opposite-signed outputs.
                generator = torch.Generator(device="cpu").manual_seed(2023)
                shape = tuple(predictions.shape[1:])
                for i in range(self.saliency_probes):
                    signs = torch.randint(0, 2, shape, generator=generator)
                    signs = signs.to(device=predictions.device, dtype=predictions.dtype) * 2 - 1
                    signs = signs.unsqueeze(0).expand_as(predictions) / math.sqrt(signs.numel())
                    gradient = torch.autograd.grad(
                        predictions, probe_x, grad_outputs=signs,
                        retain_graph=i + 1 < self.saliency_probes,
                        create_graph=False, allow_unused=True,
                    )[0]
                    if gradient is not None:
                        squared_gradients += gradient.detach().square()
            importance = torch.sqrt(squared_gradients / self.saliency_probes)
            scores = self._block_mean(importance)
            return (scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()

    def _select_blocks(self, dependency, reliability, reconstruction_error):
        # A large deviation with negligible forecasting influence is not high risk.
        eligible = (reliability < self.reliability_threshold) & (dependency >= 1.0 / self.num_chunks)
        eligible &= (reliability >= self.reliability_threshold).any(dim=1, keepdim=True)
        risk = dependency * reconstruction_error
        ranked_risk = risk.masked_fill(~eligible, -torch.inf)
        indices = ranked_risk.topk(self.top_k, dim=1).indices
        selected = torch.zeros_like(eligible).scatter_(1, indices, True)
        selected &= eligible & (risk > 0)
        return selected, risk

    def _reconstruction_loss(self, x, unreliable, anchor_std):
        reliable = ~unreliable
        # Mask 1..top_k trusted blocks, leaving at least one trusted block visible.
        number = torch.randint(1, self.top_k + 1,
                               (x.size(0), 1, x.size(2)), device=x.device)
        number = torch.minimum(number, (reliable.sum(dim=1, keepdim=True) - 1).clamp_min(0))
        priority = torch.rand(reliable.shape, device=x.device).masked_fill(~reliable, torch.inf)
        ranks = priority.argsort(dim=1).argsort(dim=1)
        training_blocks = (ranks < number) & reliable
        completion = self.imputer(x.detach(), self._point_mask(unreliable | training_blocks))
        errors = self._block_mean(((completion - x.detach()) / anchor_std).square())
        return (errors * training_blocks).sum() / training_blocks.sum().clamp_min(1)

    def forward(self, x, predictor, backbone, compute_reconstruction_loss=False):
        if x.ndim != 3 or x.size(1) != self.seq_len:
            raise ValueError("RADAR expects input shape [batch, seq_len, channels]")
        if not torch.isfinite(x).all():
            raise ValueError("RADAR expects finite observations; encode missing values before forecasting")
        original = x
        original_dtype = x.dtype
        # Detection and imputation statistics stay in float32 under mixed precision.
        x = x.float()
        with torch.no_grad():
            reliability, anchor_std = self._reliability(x)
            unreliable = reliability < self.reliability_threshold
        dependency = self._dependency(x, predictor, backbone)
        completion = self.imputer(x, self._point_mask(unreliable))
        with torch.no_grad():
            reconstruction_error = self._block_mean(((completion.detach() - x) / anchor_std).square())
            selected, risk = self._select_blocks(dependency, reliability, reconstruction_error)
            repair_mask = self._point_mask(selected)
        repaired = torch.where(repair_mask, completion.to(original_dtype), original)
        reconstruction_loss = x.new_zeros(())
        if compute_reconstruction_loss and self.training and torch.is_grad_enabled():
            reconstruction_loss = self._reconstruction_loss(x, unreliable, anchor_std)
        self.last_diagnostics = {
            "dependency": dependency,
            "reliability": reliability,
            "reconstruction_error": reconstruction_error,
            "risk": risk,
            "selected_blocks": selected,
            "repair_mask": repair_mask,
        }
        return repaired, self.reconstruction_weight * reconstruction_loss
