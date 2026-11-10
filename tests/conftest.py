from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "golden" / "synthetic"


def make_adata(seed: int = 0, n_cells: int = 300, n_genes: int = 50) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    fine = np.array(["T_CD4", "T_CD8", "B", "Mono", "NK"])
    coarse_of = {"T_CD4": "T", "T_CD8": "T", "B": "Lymph_B", "Mono": "Myeloid", "NK": "Lymph_NK"}
    ct = rng.choice(fine, n_cells)
    obs = pd.DataFrame(
        {
            "study": rng.choice(["s1", "s2", "s3"], n_cells),
            "tech": rng.choice(["10x", "smartseq"], n_cells),
            "ct_fine": ct,
            "ct_coarse": [coarse_of[c] for c in ct],
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    X = rng.poisson(rng.gamma(2.0, 1.0, (n_cells, n_genes))).astype(np.float32)
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"g{j}" for j in range(n_genes)]
    return a


@pytest.fixture
def adata():
    return make_adata()


@pytest.fixture
def golden_dir():
    if not GOLDEN.exists():
        pytest.skip("golden outputs missing; run scripts/make_golden_synthetic.py")
    return GOLDEN
