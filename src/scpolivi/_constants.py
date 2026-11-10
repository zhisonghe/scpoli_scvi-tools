"""Registry keys and constants used by scpolivi."""

from typing import NamedTuple


class _SCPOLI_REGISTRY_KEYS_NT(NamedTuple):
    # Joint (n_cells, n_condition_keys) integer codes, one embedding per column.
    CONDITIONS_KEY: str = "scpoli_conditions"
    # Joint (n_cells, n_levels) integer codes in one shared cell-type namespace; -1 = unknown.
    CELL_TYPES_KEY: str = "scpoli_cell_types"
    # (n_cells, 1) float flag, 1 = labeled cell (used by the labeled prototype loss).
    LABELED_KEY: str = "scpoli_labeled"


SCPOLI_REGISTRY_KEYS = _SCPOLI_REGISTRY_KEYS_NT()

# obs columns written by setup_anndata
COMBINED_CONDITIONS_COL = "_scpoli_conditions_combined"
LABELED_COL = "_scpoli_labeled"
INDICES_COL = "_scpoli_indices"  # row positions, used by the graph Laplacian loss

UNKNOWN_CODE = -1

# minibatch keys added by the neighbour-sampling loader of the graph Laplacian loss
GRAPH_ROLE_KEY = "scpoli_graph_role"  # 1 = anchor (all losses), 0 = sampled neighbour
GRAPH_PAIR_WEIGHT_KEY = "scpoli_graph_pair_weight"
GRAPH_GROUP_KEY = "scpoli_graph_group"
