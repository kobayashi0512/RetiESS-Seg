"""Differentiable RetiESS pooling for correlated volumetric instances.

The module treats the B-scans in one OCT volume as a bag of correlated
instances.  It rewards diagnostically useful slices while penalising several
highly similar slices from contributing the same evidence repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RetiESSOutput:
    """Outputs produced by :class:`RetiESSPool`."""

    pooled: Tensor
    weights: Tensor
    correlation: Tensor
    n_eff: Tensor
    scores: Tensor


class RetiESSPool(nn.Module):
    """Correlation-aware, entropy-regularised multiple-instance pooling.

    Args:
        feature_dim: Number of channels in each slice embedding.
        attention_hidden_dim: Hidden width of the learnable slice scorer.
        positional_mix: Weight of the positional RBF kernel in ``[0, 1]``.
            The feature-kernel weight is ``1 - positional_mix``.
        position_bandwidth: RBF bandwidth after slice positions have been
            normalised independently to ``[0, 1]`` for each volume.
        feature_bandwidth: RBF bandwidth for L2-normalised embeddings.
        redundancy_strength: Coefficient of ``w.T @ R @ w``.
        entropy_strength: Entropy coefficient. Larger values spread weights.
        num_iterations: Number of differentiable fixed-point iterations.
        damping: Update fraction in ``(0, 1]`` used for stable iteration.

    Input shape:
        ``features`` is ``[batch, slices, feature_dim]``. ``positions`` may be
        ``[slices]`` or ``[batch, slices]``. If omitted, evenly spaced slice
        indices are used. ``scores`` may optionally override the learned
        attention scorer and must have shape ``[batch, slices]``.
    """

    def __init__(
        self,
        feature_dim: int,
        attention_hidden_dim: int = 64,
        positional_mix: float = 0.5,
        position_bandwidth: float = 0.15,
        feature_bandwidth: float = 0.5,
        redundancy_strength: float = 1.0,
        entropy_strength: float = 0.25,
        num_iterations: int = 20,
        damping: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= positional_mix <= 1.0:
            raise ValueError("positional_mix must be in [0, 1]")
        if position_bandwidth <= 0.0 or feature_bandwidth <= 0.0:
            raise ValueError("RBF bandwidths must be positive")
        if redundancy_strength < 0.0 or entropy_strength <= 0.0:
            raise ValueError(
                "redundancy_strength must be non-negative and "
                "entropy_strength must be positive"
            )
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")

        self.feature_dim = feature_dim
        self.positional_mix = positional_mix
        self.position_bandwidth = position_bandwidth
        self.feature_bandwidth = feature_bandwidth
        self.redundancy_strength = redundancy_strength
        self.entropy_strength = entropy_strength
        self.num_iterations = num_iterations
        self.damping = damping

        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    @staticmethod
    def _normalise_positions(positions: Tensor) -> Tensor:
        lower = positions.amin(dim=-1, keepdim=True)
        span = positions.amax(dim=-1, keepdim=True) - lower
        return (positions - lower) / span.clamp_min(
            torch.finfo(positions.dtype).eps
        )

    def correlation_matrix(
        self, features: Tensor, positions: Tensor | None = None
    ) -> Tensor:
        """Construct the convex mixture of positional and feature RBF kernels."""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, slices, channels]")
        batch, num_slices, channels = features.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {channels}"
            )

        if positions is None:
            positions = torch.arange(
                num_slices, device=features.device, dtype=features.dtype
            ).expand(batch, -1)
        else:
            positions = positions.to(device=features.device, dtype=features.dtype)
            if positions.ndim == 1:
                if positions.shape[0] != num_slices:
                    raise ValueError("one-dimensional positions must match slices")
                positions = positions.expand(batch, -1)
            elif positions.shape != (batch, num_slices):
                raise ValueError("positions must have shape [slices] or [batch, slices]")

        positions = self._normalise_positions(positions)
        position_dist2 = (
            positions.unsqueeze(-1) - positions.unsqueeze(-2)
        ).square()
        position_kernel = torch.exp(
            -position_dist2 / (2.0 * self.position_bandwidth**2)
        )

        unit_features = F.normalize(features, p=2, dim=-1, eps=1e-12)
        feature_dist2 = torch.cdist(unit_features, unit_features).square()
        feature_kernel = torch.exp(
            -feature_dist2 / (2.0 * self.feature_bandwidth**2)
        )

        return (
            self.positional_mix * position_kernel
            + (1.0 - self.positional_mix) * feature_kernel
        )

    def _solve_weights(self, scores: Tensor, correlation: Tensor) -> Tensor:
        """Solve the entropy-regularised redundancy objective by fixed point.

        The stationary condition for

        ``max_w score.T*w - lambda/2*w.T*R*w + tau*H(w)``

        on the probability simplex is
        ``w = softmax((score - lambda*R*w) / tau)``. Unrolling a damped
        fixed-point iteration keeps the complete operation differentiable.
        """
        weights = torch.softmax(scores / self.entropy_strength, dim=-1)
        for _ in range(self.num_iterations):
            redundancy = torch.bmm(
                correlation, weights.unsqueeze(-1)
            ).squeeze(-1)
            proposal = torch.softmax(
                (scores - self.redundancy_strength * redundancy)
                / self.entropy_strength,
                dim=-1,
            )
            weights = (
                (1.0 - self.damping) * weights + self.damping * proposal
            )
        return weights

    def forward(
        self,
        features: Tensor,
        positions: Tensor | None = None,
        scores: Tensor | None = None,
    ) -> RetiESSOutput:
        correlation = self.correlation_matrix(features, positions)

        if scores is None:
            scores = self.scorer(features).squeeze(-1)
        elif scores.shape != features.shape[:2]:
            raise ValueError("scores must have shape [batch, slices]")
        scores = scores.to(device=features.device, dtype=features.dtype)

        weights = self._solve_weights(scores, correlation)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=1)
        repeated_evidence = torch.einsum(
            "bi,bij,bj->b", weights, correlation, weights
        )
        n_eff = repeated_evidence.clamp_min(
            torch.finfo(features.dtype).eps
        ).reciprocal()

        return RetiESSOutput(
            pooled=pooled,
            weights=weights,
            correlation=correlation,
            n_eff=n_eff,
            scores=scores,
        )


class ESSLogitShrinkage(nn.Module):
    """Shrink logits toward a training-set prior according to effective size.

    ``adjusted = prior + n_eff / (n_eff + kappa) * (raw - prior)``

    ``prior_logits`` must be estimated from the training split only. It may be
    a scalar (binary logit) or a vector (multiclass logits).
    """

    def __init__(self, prior_logits: Tensor | float = 0.0, kappa: float = 8.0):
        super().__init__()
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        self.register_buffer(
            "prior_logits", torch.as_tensor(prior_logits, dtype=torch.float32)
        )
        self.kappa = kappa

    def forward(self, logits: Tensor, n_eff: Tensor) -> Tensor:
        if logits.shape[0] != n_eff.shape[0]:
            raise ValueError("logits and n_eff must share their batch dimension")
        prior = self.prior_logits.to(device=logits.device, dtype=logits.dtype)
        shrinkage = n_eff / (n_eff + self.kappa)
        while shrinkage.ndim < logits.ndim:
            shrinkage = shrinkage.unsqueeze(-1)
        return prior + shrinkage * (logits - prior)
