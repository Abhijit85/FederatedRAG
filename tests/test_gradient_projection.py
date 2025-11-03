import pytest

torch = pytest.importorskip("torch")

from synapse.training.gradient import project_gradient


def test_project_gradient_scalar_case():
    lm = [torch.tensor([1.0, 2.0])]
    directions = [[torch.tensor([1.0, 0.0])]]
    lambdas = [0.5]

    projected = project_gradient(lm, directions, lambdas)
    assert torch.allclose(projected[0], torch.tensor([1.5, 2.0]), atol=1e-6)


def test_project_gradient_batch_case():
    lm = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    directions_ent = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    lambdas = [1.0]

    projected = project_gradient(lm, [directions_ent], lambdas)
    expected = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    assert projected[0].shape == lm[0].shape
    assert torch.allclose(projected[0], expected, atol=1e-6)
