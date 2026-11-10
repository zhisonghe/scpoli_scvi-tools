"""Read scArches scPoli checkpoints without importing scArches.

A scArches checkpoint directory holds ``model_params.pt`` (state dict), ``attr.pkl``
(pickled public attributes: builtins, pandas, torch tensors) and ``var_names.csv``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import torch


class _CPUUnpickler(pickle.Unpickler):
    """Unpickle tensors saved on GPU onto the CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import io

            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def read_scarches_checkpoint(dir_path: str | Path) -> tuple[dict, dict, np.ndarray]:
    """Return ``(attr_dict, state_dict, var_names)`` of a scArches scPoli checkpoint."""
    d = Path(dir_path)
    with open(d / "attr.pkl", "rb") as f:
        attrs = _CPUUnpickler(f).load()
    state = torch.load(d / "model_params.pt", map_location="cpu", weights_only=True)
    var_names = np.genfromtxt(d / "var_names.csv", delimiter=",", dtype=str)
    var_names = np.atleast_1d(var_names)
    if "conditions_" not in attrs or "recon_loss_" not in attrs:
        raise ValueError(f"{d} does not look like a scArches scPoli checkpoint.")
    return attrs, state, var_names


def module_kwargs_from_attrs(attrs: dict) -> dict:
    """SCPOLIVAE keyword arguments reproducing a scArches scPoli architecture."""
    if attrs.get("use_mmd_"):
        raise NotImplementedError("scArches models trained with use_mmd=True are not supported.")
    cond_keys = list(attrs["condition_keys_"])
    return {
        "n_input": int(attrs["input_dim_"]),
        "n_conditions_per_key": [len(attrs["conditions_"][k]) for k in cond_keys],
        "n_conditions_combined": len(attrs["conditions_combined_"]),
        "n_cell_types": len(attrs.get("cell_types_") or {}),
        "hidden_layer_sizes": list(attrs["hidden_layer_sizes_"]),
        "n_latent": int(attrs["latent_dim_"]),
        "embedding_dims": list(attrs["embedding_dims_"]),
        "embedding_max_norm": attrs["embedding_max_norm_"],
        "inject_condition": list(attrs["inject_condition_"]),
        "recon_loss": attrs["recon_loss_"],
        "dispersion": "gene-batch",
        "dropout_rate": float(attrs["dr_rate_"]),
        "use_bn": bool(attrs["use_bn_"]),
        "use_ln": bool(attrs["use_ln_"]),
        "poisson_library_scaling": False,
    }


_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^theta$"), "px_r"),
    (re.compile(r"^embeddings\.(\d+)\.weight$"), r"_embeddings_dict.cond_\1.weight"),
    # encoder
    (re.compile(r"^encoder\.FC\.L0\.expr_L\.(weight|bias)$"), r"encoder.first.expr.\1"),
    (re.compile(r"^encoder\.FC\.L0\.cond_L\.weight$"), "encoder.first.cond.weight"),
    (re.compile(r"^encoder\.FC\.N0\.(.+)$"), r"encoder.first_post.0.\1"),
    (re.compile(r"^encoder\.(mean_encoder|log_var_encoder)\.(weight|bias)$"), r"encoder.\1.\2"),
    # decoder
    (re.compile(r"^decoder\.FirstL\.L0\.expr_L\.weight$"), "decoder.first.expr.weight"),
    (re.compile(r"^decoder\.FirstL\.L0\.cond_L\.weight$"), "decoder.first.cond.weight"),
    (re.compile(r"^decoder\.FirstL\.N0\.(.+)$"), r"decoder.first_post.0.\1"),
    (re.compile(r"^decoder\.(?:mean_decoder|recon_decoder)\.0\.(weight|bias)$"), r"decoder.scale_head.\1"),
    (re.compile(r"^decoder\.dropout_decoder\.(weight|bias)$"), r"decoder.dropout_head.\1"),
]
_HIDDEN = re.compile(r"^(encoder\.FC|decoder\.HiddenL)\.([LN])(\d+)\.(.+)$")


def _map_key(key: str) -> str:
    for pat, repl in _RULES:
        if pat.match(key):
            return pat.sub(repl, key)
    m = _HIDDEN.match(key)
    if m:
        part, kind, idx, rest = m.groups()
        owner = "encoder" if part.startswith("encoder") else "decoder"
        sub = 0 if kind == "L" else 1
        return f"{owner}.hidden.{int(idx) - 1}.{sub}.{rest}"
    raise KeyError(f"Unrecognized scArches parameter {key!r}.")


def split_combined(combined: str, values_per_key: list[list[str]]) -> list[int] | None:
    """Per-key codes of a combined condition ``v1_v2_..`` (values may contain '_')."""
    if not values_per_key:
        return [] if combined == "" else None
    first, rest = values_per_key[0], values_per_key[1:]
    for i, v in enumerate(first):
        v = str(v)
        if not rest and combined == v:
            return [i]
        if rest and combined.startswith(v + "_"):
            tail = split_combined(combined[len(v) + 1 :], rest)
            if tail is not None:
                return [i, *tail]
    return None


def combined_to_conditions(attrs: dict) -> torch.Tensor:
    """(n_combined, n_keys) codes for each scArches combined condition."""
    keys = list(attrs["condition_keys_"])
    values = [list(attrs["conditions_"][k]) for k in keys]
    rows = []
    for c in attrs["conditions_combined_"]:
        codes = split_combined(str(c), values)
        if codes is None:
            raise ValueError(f"Cannot split combined condition {c!r} into {keys}.")
        rows.append(codes)
    return torch.tensor(rows, dtype=torch.long).reshape(len(rows), len(keys))


def convert_state_dict(state: dict, attrs: dict) -> dict:
    """Map a scArches scPoli state dict (and stored prototypes) to SCPOLIVAE names."""
    out = {_map_key(k): v.clone() for k, v in state.items()}
    out["combined_to_conditions"] = combined_to_conditions(attrs)
    latent = int(attrs["latent_dim_"])
    n_ct = len(attrs.get("cell_types_") or {})
    lab = attrs.get("prototypes_labeled_") or {}
    unl = attrs.get("prototypes_unlabeled_") or {}
    mean = torch.full((n_ct, latent), float("nan"))
    cv = torch.full((n_ct, latent, latent), float("nan"))
    if lab.get("mean") is not None:
        m = torch.as_tensor(lab["mean"]).detach().float().cpu()
        mean[: m.shape[0]] = m
        c = torch.as_tensor(lab["cov"]).detach().float().cpu().reshape(-1, latent, latent)
        cv[: c.shape[0]] = c
    out["prototypes_labeled_mean"] = mean
    out["prototypes_labeled_cov"] = cv
    out["prototypes_fixed"] = torch.zeros(n_ct, dtype=torch.bool)
    if unl.get("mean") is not None:
        u = torch.as_tensor(unl["mean"]).detach().float().cpu()
        out["prototypes_unlabeled_mean"] = u.reshape(-1, latent)
    else:
        out["prototypes_unlabeled_mean"] = torch.zeros(0, latent)
    return out
