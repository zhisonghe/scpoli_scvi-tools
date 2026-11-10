import numpy as np
import pytest
import torch

from scpolivi.module import _prototypes as P


@pytest.fixture
def g(golden_dir):
    return dict(np.load(golden_dir / "prototype_functions.npz"))


@pytest.mark.parametrize("p", [1, 2])
def test_labeled_loss_matches_scarches(g, p):
    z, protos, labels = (torch.tensor(g[k]) for k in ("latent", "protos", "labels"))
    ours = P.labeled_prototype_loss(z, protos, labels, p=float(p))
    np.testing.assert_allclose(ours.numpy(), g[f"labeled_loss_p{p}"], rtol=1e-5)
    loop = P._labeled_prototype_loss_loop(z, protos, labels, p=float(p))
    np.testing.assert_allclose(ours.numpy(), loop.numpy(), rtol=1e-5)


@pytest.mark.parametrize("p", [1, 2])
def test_unlabeled_loss_matches_scarches(g, p):
    z, protos = torch.tensor(g["latent"]), torch.tensor(g["protos_unlab"])
    loss, counts = P.unlabeled_prototype_loss(z, protos, p=float(p))
    np.testing.assert_allclose(loss.numpy(), g[f"unlabeled_loss_p{p}"], rtol=1e-5)
    np.testing.assert_array_equal(counts.numpy(), g[f"unlabeled_counts_p{p}"])


def test_update_matches_scarches(g):
    z, labels = torch.tensor(g["latent"]), torch.tensor(g["labels"])
    pm, pc = torch.tensor(g["protos"]), torch.tensor(g["protos_cov"])
    m, c = P.update_labeled_prototypes(z, labels, pm, pc, n_classes=6)
    np.testing.assert_allclose(m.numpy(), g["update_mean_all"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c.numpy(), g["update_cov_all"], rtol=1e-5, atol=1e-6)
    mask = torch.zeros(6, dtype=torch.bool)
    mask[[4, 5]] = True
    m, c = P.update_labeled_prototypes(z, labels, pm, pc, n_classes=6, update_mask=mask)
    np.testing.assert_allclose(m.numpy(), g["update_mean_mask45"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c.numpy(), g["update_cov_mask45"], rtol=1e-5, atol=1e-6)


def test_labeled_loss_all_unknown():
    z = torch.randn(5, 3)
    assert P.labeled_prototype_loss(z, torch.randn(2, 3), -torch.ones(5, 2, dtype=torch.long)) == 0


def test_classify_restricts_to_classes():
    protos = torch.tensor([[0.0, 0.0], [10.0, 10.0], [0.1, 0.1]])
    z = torch.tensor([[0.09, 0.09], [9.0, 9.0]])
    pred, unc, _ = P.classify(z, protos, torch.tensor([0, 1]))
    assert pred.tolist() == [0, 1]
    pred, _, prob = P.classify(z, protos, torch.tensor([1, 2]), get_prob=True)
    assert pred.tolist() == [2, 1] and torch.allclose(prob.sum(1), torch.ones(2))
