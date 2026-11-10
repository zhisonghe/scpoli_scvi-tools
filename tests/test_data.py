import numpy as np
import pandas as pd
import pytest
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager

from scpolivi._constants import COMBINED_CONDITIONS_COL, LABELED_COL, SCPOLI_REGISTRY_KEYS as K
from scpolivi.data import build_fields, write_helper_columns


def register(adata, cond=("study", "tech"), cts=("ct_fine", "ct_coarse"), unknown=None,
             labeled=None, source=None, **transfer):
    write_helper_columns(adata, list(cond), labeled, keep_existing_labeled=source is not None)
    mgr = AnnDataManager(fields=build_fields(None, list(cond), list(cts), unknown))
    if source is None:
        mgr.register_fields(adata)
    else:
        mgr.register_fields(adata, source_registry=source.registry, **transfer)
    return mgr


def test_conditions_appearance_order_and_combined(adata):
    mgr = register(adata)
    st = mgr.get_state_registry(K.CONDITIONS_KEY)
    assert list(st["mappings"]["study"]) == list(pd.unique(adata.obs["study"].astype(str)))
    assert list(st["n_cats_per_key"]) == [3, 2]
    codes = mgr.get_from_registry(K.CONDITIONS_KEY)
    assert codes.shape == (adata.n_obs, 2)
    comb = mgr.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["categorical_mapping"]
    assert len(comb) == 6 and all("_" in c for c in comb)
    assert (adata.obs[COMBINED_CONDITIONS_COL].astype(str)
            == adata.obs.study.astype(str) + "_" + adata.obs.tech.astype(str)).all()


def test_shared_namespace_and_unknown(adata):
    adata.obs.loc[adata.obs_names[:10], "ct_fine"] = "unknown"
    mgr = register(adata, unknown=["unknown"])
    st = mgr.get_state_registry(K.CELL_TYPES_KEY)
    names = st["mapping"]
    assert "unknown" not in names
    # fine names first (level order), then coarse names; "T" etc. on coarse level only
    assert set(names) == {"T_CD4", "T_CD8", "B", "Mono", "NK", "T", "Lymph_B", "Myeloid", "Lymph_NK"}
    assert list(st["cell_types"]["T"]) == ["ct_coarse"]
    codes = np.asarray(mgr.get_from_registry(K.CELL_TYPES_KEY))
    assert codes.shape == (adata.n_obs, 2)
    assert (codes[:10, 0] == -1).all()
    assert (codes[:, 1] >= 0).all()
    assert mgr.summary_stats[f"n_{K.CELL_TYPES_KEY}"] == 9


def test_same_name_two_levels_shares_code(adata):
    adata.obs["ct_coarse"] = adata.obs["ct_coarse"].replace({"Lymph_NK": "NK"})
    mgr = register(adata)
    st = mgr.get_state_registry(K.CELL_TYPES_KEY)
    assert list(st["cell_types"]["NK"]) == ["ct_fine", "ct_coarse"]
    codes = np.asarray(mgr.get_from_registry(K.CELL_TYPES_KEY))
    nk = list(st["mapping"]).index("NK")
    is_nk = (adata.obs.ct_fine == "NK").values
    assert (codes[is_nk, 0] == nk).all() and (codes[is_nk, 1] == nk).all()


def test_labeled_indices_restrict_namespace(adata):
    labeled = np.where(adata.obs.ct_fine != "NK")[0]
    mgr = register(adata, labeled=labeled)
    st = mgr.get_state_registry(K.CELL_TYPES_KEY)
    assert "NK" not in st["mapping"]
    flag = np.asarray(mgr.get_from_registry(K.LABELED_KEY)).ravel()
    assert flag.sum() == len(labeled)
    # unlabeled NK cells get -1 at the fine level
    codes = np.asarray(mgr.get_from_registry(K.CELL_TYPES_KEY))
    assert (codes[(adata.obs.ct_fine == "NK").values, 0] == -1).all()


def test_query_transfer_extends(adata):
    ref = adata[adata.obs.study != "s3"].copy()
    ref = ref[ref.obs.ct_fine != "NK"].copy()
    qry = adata[adata.obs.study == "s3"].copy()
    ref_mgr = register(ref)
    qry.obs[LABELED_COL] = (qry.obs.ct_fine == "NK").astype(np.float32).values
    q = register(qry, source=ref_mgr, extend_categories=True, allow_missing_labels=True)
    cst = q.get_state_registry(K.CONDITIONS_KEY)
    assert list(cst["mappings"]["study"])[:2] == list(
        ref_mgr.get_state_registry(K.CONDITIONS_KEY)["mappings"]["study"])
    assert "s3" in cst["mappings"]["study"]
    lst = q.get_state_registry(K.CELL_TYPES_KEY)
    ref_names = ref_mgr.get_state_registry(K.CELL_TYPES_KEY)["mapping"]
    assert list(lst["mapping"])[: len(ref_names)] == list(ref_names)
    assert "NK" in lst["mapping"] and list(lst["mapping"]).index("NK") >= len(ref_names)
    # combined batch categories extended too
    assert len(q.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["categorical_mapping"]) == 6


def test_query_missing_label_columns(adata):
    ref = adata[adata.obs.study != "s3"].copy()
    qry = adata[adata.obs.study == "s3"].copy()
    del qry.obs["ct_fine"], qry.obs["ct_coarse"]
    ref_mgr = register(ref)
    with pytest.warns(UserWarning, match="Missing cell type keys"):
        q = register(qry, source=ref_mgr, extend_categories=True, allow_missing_labels=True)
    assert (np.asarray(q.get_from_registry(K.CELL_TYPES_KEY)) == -1).all()


def test_no_cell_type_keys(adata):
    mgr = register(adata, cts=())
    assert mgr.summary_stats[f"n_{K.CELL_TYPES_KEY}"] == 0
