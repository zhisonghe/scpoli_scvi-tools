"""Observation models for the scPoli losses that scvi-tools does not ship.

They follow the scvi-tools distribution interface used by the RNA-seq mixins
(``log_prob``, ``mean``, ``sample``, ``scale``, ``get_normalized``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints


class SquaredError(Distribution):
    """Pseudo-distribution whose ``log_prob`` is minus the squared error (scArches ``mse``)."""

    arg_constraints = {"loc": constraints.real}
    support = constraints.real

    def __init__(self, loc: torch.Tensor, validate_args: bool = False):
        self.loc = loc
        self.scale = loc
        super().__init__(batch_shape=loc.shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -F.mse_loss(self.loc, value, reduction="none")

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # noqa: B008
        return self.loc.expand(torch.Size(sample_shape) + self.loc.shape).clone()

    def get_normalized(self, key) -> torch.Tensor:
        return self.loc


class BinarizedBernoulli(Distribution):
    """Bernoulli on ``x > 0`` (scArches ``bernoulli``: BCE against binarized data)."""

    arg_constraints = {"probs": constraints.unit_interval}
    support = constraints.real

    def __init__(self, probs: torch.Tensor, validate_args: bool = False):
        self.probs = probs
        self.scale = probs
        super().__init__(batch_shape=probs.shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.probs

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -F.binary_cross_entropy(self.probs, (value > 0).float(), reduction="none")

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # noqa: B008
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(torch.Size(sample_shape) + self.probs.shape))

    def get_normalized(self, key) -> torch.Tensor:
        return self.probs
