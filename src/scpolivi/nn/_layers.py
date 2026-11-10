"""Encoder/decoder networks of scPoli.

Layer order, biases, normalization settings and heads mirror scArches' scPoli
(``scarches/models/scpoli/scpoli.py``) so that scArches checkpoints map 1:1 onto these
modules. scvi-tools' ``FCLayers`` is not used because scPoli allows a different width per
hidden layer and injects conditions through a separate bias-free linear map.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

RECON_LOSSES = ("nb", "zinb", "nb_dist", "poisson", "mse", "bernoulli")
ReconLoss = Literal["nb", "zinb", "nb_dist", "poisson", "mse", "bernoulli"]


class CondLinear(nn.Module):
    """``W x + C c (+ b)``: linear map of the input plus a bias-free map of the condition embedding."""

    def __init__(self, n_in: int, n_out: int, n_cond: int, bias: bool):
        super().__init__()
        self.n_cond = n_cond
        self.expr = nn.Linear(n_in, n_out, bias=bias)
        self.cond = nn.Linear(n_cond, n_out, bias=False) if n_cond > 0 else None

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        out = self.expr(x)
        if self.cond is not None:
            out = out + self.cond(c)
        return out


def _post_layers(n_out: int, use_bn: bool, use_ln: bool, dropout_rate: float) -> list[nn.Module]:
    mods: list[nn.Module] = []
    if use_bn:
        mods.append(nn.BatchNorm1d(n_out, affine=True))
    elif use_ln:
        mods.append(nn.LayerNorm(n_out, elementwise_affine=False))
    mods.append(nn.ReLU())
    if dropout_rate > 0:
        mods.append(nn.Dropout(p=dropout_rate))
    return mods


class SCPoliEncoder(nn.Module):
    """x -> (mean, log_var). Conditions enter the first layer only."""

    def __init__(
        self,
        n_input: int,
        hidden_layer_sizes: list[int],
        n_latent: int,
        n_cond: int,
        use_bn: bool,
        use_ln: bool,
        dropout_rate: float,
    ):
        super().__init__()
        sizes = [n_input, *hidden_layer_sizes]
        self.first = None
        self.first_post = None
        self.hidden = nn.ModuleList()
        if len(sizes) > 1:
            self.first = CondLinear(sizes[0], sizes[1], n_cond, bias=True)
            self.first_post = nn.Sequential(*_post_layers(sizes[1], use_bn, use_ln, dropout_rate))
            for n_in, n_out in zip(sizes[1:-1], sizes[2:], strict=True):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(n_in, n_out, bias=True),
                        *_post_layers(n_out, use_bn, use_ln, dropout_rate),
                    )
                )
        self.mean_encoder = nn.Linear(sizes[-1], n_latent)
        self.log_var_encoder = nn.Linear(sizes[-1], n_latent)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None):
        if self.first is not None:
            x = self.first_post(self.first(x, c))
            for layer in self.hidden:
                x = layer(x)
        return self.mean_encoder(x), self.log_var_encoder(x)


class SCPoliDecoder(nn.Module):
    """z -> likelihood parameters. Bias-free hidden layers, as in scArches."""

    def __init__(
        self,
        n_latent: int,
        hidden_layer_sizes: list[int],
        n_output: int,
        n_cond: int,
        recon_loss: ReconLoss,
        use_bn: bool,
        use_ln: bool,
        dropout_rate: float,
    ):
        super().__init__()
        if len(hidden_layer_sizes) == 0:
            raise ValueError("scPoli requires at least one hidden layer.")
        self.recon_loss = recon_loss
        rev = list(reversed(hidden_layer_sizes))
        self.first = CondLinear(n_latent, rev[0], n_cond, bias=False)
        self.first_post = nn.Sequential(*_post_layers(rev[0], use_bn, use_ln, dropout_rate))
        self.hidden = nn.ModuleList(
            nn.Sequential(
                nn.Linear(n_in, n_out, bias=False),
                *_post_layers(n_out, use_bn, use_ln, dropout_rate),
            )
            for n_in, n_out in zip(rev[:-1], rev[1:], strict=True)
        )
        n_last = rev[-1]
        if recon_loss in ("nb", "nb_dist", "zinb", "poisson"):
            self.scale_head = nn.Linear(n_last, n_output)
            self.activation = nn.Softmax(dim=-1)
        elif recon_loss == "mse":
            self.scale_head = nn.Linear(n_last, n_output)
            self.activation = nn.ReLU()
        elif recon_loss == "bernoulli":
            self.scale_head = nn.Linear(n_last, n_output)
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"recon_loss must be one of {RECON_LOSSES}, got {recon_loss!r}.")
        self.dropout_head = nn.Linear(n_last, n_output) if recon_loss == "zinb" else None

    def forward(self, z: torch.Tensor, c: torch.Tensor | None):
        h = self.first_post(self.first(z, c))
        for layer in self.hidden:
            h = layer(h)
        scale = self.activation(self.scale_head(h))
        dropout_logits = self.dropout_head(h) if self.dropout_head is not None else None
        return scale, dropout_logits
