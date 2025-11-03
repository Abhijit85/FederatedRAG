from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def _flatten_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return tensor.reshape(1, -1) if tensor.ndim == 1 else tensor.reshape(1, 1)
    batch = tensor.shape[0]
    return tensor.reshape(batch, -1)


def project_gradient(
    gradients: Sequence[torch.Tensor | None],
    directions_list: Iterable[Sequence[torch.Tensor | None]],
    lambdas: Sequence[float],
    eps: float = 1e-8,
) -> List[torch.Tensor | None]:
    """
    Project the base gradients onto semantic directions.

    Args:
        gradients: sequence of base gradients (LM) for each parameter. May contain None.
        directions_list: iterable whose elements are sequences matching `gradients` in length.
            Each sequence contains the semantic direction tensors for one objective.
        lambdas: weights applied to each projection.
        eps: numerical stability constant.

    Returns:
        List of tensors representing the steered gradient for each parameter (None preserved).
    """
    projected: List[torch.Tensor | None] = []
    dir_sequences = list(directions_list)

    for idx, grad in enumerate(gradients):
        if grad is None:
            projected.append(None)
            continue

        g_new = grad.clone()
        for lam, dir_seq in zip(lambdas, dir_sequences):
            if idx >= len(dir_seq):
                continue
            direction = dir_seq[idx]
            if direction is None:
                continue

            if direction.ndim >= 2 and grad.ndim >= 2 and direction.shape[0] == grad.shape[0]:
                # Batched (per-sample) gradients
                grad_flat = _flatten_except_batch(grad)
                dir_flat = _flatten_except_batch(direction)

                denom = (dir_flat * dir_flat).sum(dim=-1, keepdim=True).clamp_min(eps)
                dot = (grad_flat * dir_flat).sum(dim=-1, keepdim=True)
                coeff = dot / denom

                view_shape = [direction.shape[0]] + [1] * (direction.ndim - 1)
                g_new = g_new + lam * coeff.view(*view_shape) * direction
            else:
                denom = torch.sum(direction * direction).clamp_min(eps)
                dot = torch.sum(grad * direction)
                coeff = dot / denom
                g_new = g_new + lam * coeff * direction

        projected.append(g_new)
    return projected
