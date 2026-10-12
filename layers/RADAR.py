"""RADAR v2.1: confidence-gated risk scoring and soft residual suppression."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def geometric_chunk_lengths(seq_len):
    """Return deterministic chunk lengths, ordered from newest to oldest."""
    if not isinstance(seq_len, int) or seq_len < 1:
        raise ValueError("RADAR seq_len must be a positive integer")
    if seq_len < 8:
        return (1,) * seq_len

    template = (2, 2, 4, 4, 8, 16, 28, 32)
    lengths = [max(1, seq_len * length // 96) for length in template]
    remainder = seq_len - sum(lengths)
    if remainder > 0:
        order = sorted(
            range(8),
            key=lambda index: (seq_len * template[index] % 96, index),
            reverse=True,
        )
        for index in order[:remainder]:
            lengths[index] += 1
    while sum(lengths) > seq_len:
        index = max(range(8), key=lambda candidate: (lengths[candidate], candidate))
        lengths[index] -= 1
    return tuple(sorted(lengths))


class RADAR(nn.Module):
    """Filter chunks whose risk exceeds a confidence threshold.

    Risk combines a fixed recency prior with a local, trend-aware Gaussian
    distribution comparison. Gated chunks keep their own mean and only have
    their residual amplitude reduced. The module has no trainable parameters,
    ranking step, masks, or imputer.
    """

    def __init__(self, seq_len, risk_threshold=0.75, receptive_field=8,
                 anchor_window=32,
                 dependency_decay=3.0, gamma=1.0, mean_weight=0.01,
                 variance_weight=1.0, repair_strength=10.0,
                 store_diagnostics=False):
        super().__init__()
        if not isinstance(receptive_field, int) or receptive_field < 1:
            raise ValueError("RADAR receptive_field must be a positive integer")
        if not isinstance(anchor_window, int) or anchor_window < 1:
            raise ValueError("RADAR anchor_window must be a positive integer")
        scalar_settings = {
            "risk_threshold": risk_threshold,
            "dependency_decay": dependency_decay,
            "gamma": gamma,
            "mean_weight": mean_weight,
            "variance_weight": variance_weight,
            "repair_strength": repair_strength,
        }
        try:
            settings_are_finite = all(
                math.isfinite(float(value)) for value in scalar_settings.values())
        except (TypeError, ValueError, OverflowError):
            settings_are_finite = False
        if not settings_are_finite:
            raise ValueError("RADAR scalar settings must be finite real numbers")
        if not 0 <= risk_threshold <= 1:
            raise ValueError("RADAR risk_threshold must be between 0 and 1")
        if dependency_decay <= 0 or gamma <= 0 or repair_strength <= 0:
            raise ValueError("RADAR decay, gamma, and repair strength must be positive")
        if mean_weight < 0 or variance_weight <= 0:
            raise ValueError("RADAR statistical weights are invalid")

        self.seq_len = seq_len
        self.chunk_lengths = geometric_chunk_lengths(seq_len)
        self.num_chunks = len(self.chunk_lengths)
        self.risk_threshold = float(risk_threshold)
        self.receptive_field = receptive_field
        self.anchor_window = anchor_window
        self.dependency_decay = float(dependency_decay)
        self.gamma = float(gamma)
        self.mean_weight = float(mean_weight)
        self.variance_weight = float(variance_weight)
        self.repair_strength = float(repair_strength)
        self.store_diagnostics = bool(store_diagnostics)

        chunk_bounds = []
        start = 0
        for length in reversed(self.chunk_lengths):
            chunk_bounds.append((start, start + length))
            start += length
        self.chunk_bounds = tuple(chunk_bounds)

        statistic_bounds = []
        for start, end in self.chunk_bounds:
            if end - start < self.receptive_field:
                statistic_bounds.append((max(0, end - self.receptive_field), end))
            else:
                statistic_bounds.append((start, end))
        self.statistic_bounds = tuple(statistic_bounds)
        self.anchor_bounds = tuple(
            (max(0, start - self.anchor_window), start)
            for start, _ in self.statistic_bounds
        )

        self._detrend_buffer_names = {}
        self._detrend_scales = {}
        window_lengths = {
            end - start for start, end in self.statistic_bounds + self.anchor_bounds
            if end - start > 1
        }
        for length in sorted(window_lengths):
            positions = torch.arange(length, dtype=torch.float32)
            positions = positions - positions.mean()
            name = f"_detrend_positions_{length}"
            self.register_buffer(name, positions, persistent=False)
            self._detrend_buffer_names[length] = name
            self._detrend_scales[length] = 1.0 / positions.square().sum().item()

        block_ids = torch.empty(seq_len, dtype=torch.long)
        for index, (start, end) in enumerate(self.chunk_bounds):
            block_ids[start:end] = index
        membership = F.one_hot(block_ids, self.num_chunks).float()

        gaps = torch.tensor(
            [seq_len - end for _, end in self.chunk_bounds], dtype=torch.float32)
        dependency = torch.exp(-self.dependency_decay * gaps / float(seq_len))
        anchor_valid = torch.tensor(
            [end > start for start, end in self.anchor_bounds], dtype=torch.bool)
        local_centers = torch.tensor(
            [(start + end - 1) / 2 for start, end in self.statistic_bounds],
            dtype=torch.float32,
        )
        anchor_centers = torch.tensor(
            [((start + end - 1) / 2 if end > start else local_centers[index].item())
             for index, (start, end) in enumerate(self.anchor_bounds)],
            dtype=torch.float32,
        )

        self.register_buffer("block_ids", block_ids, persistent=False)
        self.register_buffer("membership", membership, persistent=False)
        self.register_buffer("block_sizes", membership.sum(dim=0), persistent=False)
        self.register_buffer("dependency_prior", dependency, persistent=False)
        self.register_buffer("anchor_valid_prior", anchor_valid, persistent=False)
        self.register_buffer(
            "anchor_projection_offsets", local_centers - anchor_centers,
            persistent=False,
        )
        self.last_diagnostics = None

    def set_diagnostics(self, enabled=True):
        """Enable or disable detailed per-batch diagnostics.

        Diagnostics are disabled by default because retaining all per-block GPU
        tensors can be expensive for high-dimensional datasets.
        """
        self.store_diagnostics = bool(enabled)
        if not self.store_diagnostics:
            self.last_diagnostics = None
        return self

    def _block_mean(self, values):
        totals = torch.einsum(
            "blc,lk->bkc", values, self.membership.to(values.dtype))
        return totals / self.block_sizes.to(values.dtype).view(1, -1, 1)

    def _window_statistics(self, values, bounds, include_slopes=False):
        means = []
        standard_deviations = []
        residual_standard_deviations = []
        slopes = []
        roundoff_multiplier = 64 * torch.finfo(values.dtype).eps

        for start, end in bounds:
            if end <= start:
                zeros = values.new_zeros(values.size(0), values.size(2))
                means.append(zeros)
                standard_deviations.append(zeros)
                residual_standard_deviations.append(zeros)
                if include_slopes:
                    slopes.append(zeros)
                continue

            window = values[:, start:end, :]
            mean = window.mean(dim=1)
            centered = window - mean.unsqueeze(1)
            variance = centered.square().mean(dim=1)
            standard_deviation = torch.sqrt(variance)

            if end - start > 1:
                length = end - start
                positions = getattr(
                    self, self._detrend_buffer_names[length]).to(values.dtype)
                slope = (centered * positions.view(1, -1, 1)).sum(dim=1)
                slope = slope * self._detrend_scales[length]
                residual = centered - slope.unsqueeze(1) * positions.view(1, -1, 1)
                residual_variance = residual.square().mean(dim=1)

                # Least-squares roundoff scales with the signal variance. A
                # relative floor preserves the score under changes of units.
                residual_variance = torch.where(
                    residual_variance <= roundoff_multiplier * variance,
                    torch.zeros_like(residual_variance),
                    residual_variance,
                )
                residual_standard_deviation = torch.sqrt(residual_variance)
            else:
                slope = torch.zeros_like(standard_deviation)
                residual_standard_deviation = torch.zeros_like(standard_deviation)

            means.append(mean)
            standard_deviations.append(standard_deviation)
            residual_standard_deviations.append(residual_standard_deviation)
            if include_slopes:
                slopes.append(slope)

        statistics = (
            torch.stack(means, dim=1),
            torch.stack(standard_deviations, dim=1),
            torch.stack(residual_standard_deviations, dim=1),
        )
        if include_slopes:
            return statistics + (torch.stack(slopes, dim=1),)
        return statistics

    @staticmethod
    def _scale_free_ratio(numerator, denominator):
        """Divide same-unit squared quantities without an absolute data floor."""
        tiny = torch.finfo(numerator.dtype).tiny
        ratio = numerator / denominator.clamp_min(tiny)
        return torch.where(denominator > 0, ratio, torch.zeros_like(ratio))

    def _point_mask(self, blocks):
        return blocks[:, self.block_ids, :]

    def _score(self, values, detailed=False):
        local_statistics = self._window_statistics(
            values, self.statistic_bounds, include_slopes=detailed)
        local_mean, local_std, local_residual_std = local_statistics[:3]
        local_slope = local_statistics[3] if detailed else None
        anchor_mean, anchor_std, anchor_residual_std, anchor_slope = \
            self._window_statistics(values, self.anchor_bounds, include_slopes=True)
        anchor_valid = self.anchor_valid_prior.view(1, -1, 1)

        # Compare the local mean with the preceding trend extrapolated to the
        # local receptive-field center. This makes every affine trend a zero-
        # risk reference regardless of its slope or measurement unit.
        projection_offsets = self.anchor_projection_offsets.to(values.dtype)
        projected_anchor_mean = (
            anchor_mean + anchor_slope * projection_offsets.view(1, -1, 1))

        mean_wasserstein_sq = (local_mean - projected_anchor_mean).square()

        # Reliability is deliberately directional in the variance component:
        # a sudden increase in detrended residual variance is evidence of a
        # high-frequency burst, while a quieter recent window is not.
        pooled_raw_variance = 0.5 * (
            local_std.square() + anchor_std.square())
        mean_distance = self._scale_free_ratio(
            mean_wasserstein_sq,
            mean_wasserstein_sq + pooled_raw_variance,
        )
        variance_burst_sq = torch.relu(
            local_residual_std - anchor_residual_std).square()
        variance_distance = self._scale_free_ratio(
            variance_burst_sq, pooled_raw_variance)
        mean_distance = torch.where(
            anchor_valid, mean_distance, torch.zeros_like(mean_distance))
        variance_distance = torch.where(
            anchor_valid, variance_distance, torch.zeros_like(variance_distance))
        risk_distance = (
            self.mean_weight * mean_distance
            + self.variance_weight * variance_distance
        )
        reliability = torch.exp(-self.gamma * risk_distance)
        reliability = torch.where(
            anchor_valid, reliability, torch.ones_like(reliability))

        dependency = self.dependency_prior.to(values.dtype).view(1, -1, 1)
        risk = dependency * (1 - reliability)
        if not detailed:
            return risk

        # Fill the undefined oldest Anchor only for readable diagnostics. Its
        # reliability has already been forced to one above.
        anchor_mean = torch.where(anchor_valid, anchor_mean, local_mean)
        projected_anchor_mean = torch.where(
            anchor_valid, projected_anchor_mean, local_mean)
        anchor_std = torch.where(anchor_valid, anchor_std, local_std)
        anchor_residual_std = torch.where(
            anchor_valid, anchor_residual_std, local_residual_std)
        anchor_slope = torch.where(anchor_valid, anchor_slope, local_slope)
        mean_wasserstein_sq = (local_mean - projected_anchor_mean).square()
        scale_wasserstein_sq = (local_residual_std - anchor_residual_std).square()
        wasserstein_sq = mean_wasserstein_sq + scale_wasserstein_sq
        dependency = dependency.expand(values.size(0), -1, values.size(2))
        return {
            "dependency": dependency,
            "reliability": reliability,
            "risk": risk,
            "risk_distance": risk_distance,
            "wasserstein_distance": torch.sqrt(wasserstein_sq),
            "wasserstein_squared": wasserstein_sq,
            "mean_distance": mean_distance,
            "variance_distance": variance_distance,
            "local_mean": local_mean,
            "local_std": local_std,
            "local_residual_std": local_residual_std,
            "local_slope": local_slope,
            "anchor_mean": anchor_mean,
            "anchor_projected_mean": projected_anchor_mean,
            "anchor_std": anchor_std,
            "anchor_residual_std": anchor_residual_std,
            "anchor_slope": anchor_slope,
            "anchor_valid": anchor_valid.expand_as(reliability),
        }

    def _gate(self, risk):
        return risk > self.risk_threshold

    @staticmethod
    def _require_finite(x):
        finite = torch.isfinite(x).all()
        if x.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(finite, "RADAR requires finite observations")
        elif not finite.item():
            raise ValueError("RADAR requires finite observations; found NaN or Inf")

    def forward(self, x):
        if x.ndim != 3 or x.size(1) != self.seq_len:
            raise ValueError("RADAR expects input shape [batch, seq_len, channels]")
        if not torch.is_floating_point(x):
            raise TypeError("RADAR expects floating-point observations")
        self._require_finite(x)

        original_dtype = x.dtype
        statistics = x if x.dtype in (torch.float32, torch.float64) else x.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            with torch.no_grad():
                score = self._score(statistics, detailed=self.store_diagnostics)
                diagnostics = score if self.store_diagnostics else None
                risk = diagnostics["risk"] if diagnostics is not None else score
                selected = self._gate(risk)
                block_alpha = torch.exp(
                    -self.repair_strength * risk.square())
                block_alpha = torch.where(
                    selected, block_alpha, torch.ones_like(block_alpha))
                repair_mask = self._point_mask(selected)

            trend = self._block_mean(statistics)
            point_trend = trend[:, self.block_ids, :]
            point_alpha = block_alpha[:, self.block_ids, :]
            softened = point_trend + point_alpha * (statistics - point_trend)
        repaired = torch.where(repair_mask, softened.to(original_dtype), x)

        if self.store_diagnostics:
            self.last_diagnostics = {
                **{name: value.detach() for name, value in diagnostics.items()},
                "alpha": block_alpha.detach(),
                "trend": trend.detach(),
                "selected_blocks": selected.detach(),
                "repair_mask": repair_mask.detach(),
            }
        else:
            self.last_diagnostics = None
        return repaired
