from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional
    torch = None
    nn = object  # type: ignore
    F = None  # type: ignore

from synapse.training.gradient import project_gradient
from synapse.training.lora import LoRALayerConfig
from synapse.training.texgrad import TexGradSample, TexGradHead, TexGradConfig
from synapse.training.texgrad_models import CitationAligner, EntailmentScorer


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for TexGrad backpropagation. "
            "Install torch/accelerate/peft in your environment."
        )


def _hash_tokens(text: str, dim: int) -> torch.Tensor:
    """
    Lightweight hashed embedding to convert text into a fixed-size vector.
    Deterministic and differentiable when treated as constant input.
    """
    ids = []
    for token in text.split():
        hashed = hash(token) % dim
        ids.append(hashed)
    vec = torch.zeros(dim, dtype=torch.float32)
    for idx in ids:
        vec[idx] += 1.0
    if vec.norm() > 0:
        vec = vec / vec.norm()
    return vec


class _LoRALinear(nn.Module):  # pragma: no cover - requires torch
    """
    Minimal LoRA module used to simulate adapter training on top of a frozen weight.
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.base_weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01, requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(output_dim, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_B @ self.lora_A
        weight = self.base_weight + delta
        return x @ weight.t()

    def lora_delta(self) -> torch.Tensor:
        with torch.no_grad():
            return (self.lora_B @ self.lora_A).detach().clone()


@dataclass
class TexGradBatch:
    """
    Batched representation of examples to feed the trainer.
    """

    question: str
    answer: str
    positives: Sequence[str]
    negatives: Sequence[str]


class TexGradLoRATrainer:
    """
    Implements a differentiable TexGrad training loop over synthetic LoRA modules.

    This provides real backpropagation behaviour without loading a full LLM,
    making it practical for unit tests or CPU-only environments. In production,
    swap this module for a PEFT+transformers integration.
    """

    def __init__(
        self,
        lora_config: LoRALayerConfig,
        texgrad_config: TexGradConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        _require_torch()
        self.config = lora_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.texgrad_config = texgrad_config or TexGradConfig()
        self.texgrad_head = TexGradHead(self.texgrad_config)
        self.entail_scorer = EntailmentScorer(device=self.device)
        self.citation_aligner = CitationAligner(device=self.device)

        base_dim = self.config.base_dimension
        self.modules: Dict[str, _LoRALinear] = {}
        for layer in self.config.resolve_layers():
            module = _LoRALinear(base_dim, base_dim, rank=max(self.config.rank_choices))
            self.modules[layer] = module.to(self.device)

        self.optimizer = torch.optim.Adam(
            (param for module in self.modules.values() for param in module.parameters() if param.requires_grad),
            lr=1e-3,
        )
        self.rank = max(self.config.rank_choices)

    def _encode_sample(self, sample: TexGradSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = self.config.base_dimension
        q_vec = _hash_tokens(sample.question, dim).to(self.device)
        a_vec = _hash_tokens(sample.answer, dim).to(self.device)
        pos_vec = sum((_hash_tokens(text, dim) for text in sample.positive_contexts), torch.zeros(dim)) / max(
            len(sample.positive_contexts), 1
        )
        neg_vec = sum((_hash_tokens(text, dim) for text in sample.negative_contexts), torch.zeros(dim)) / max(
            len(sample.negative_contexts), 1
        )
        return q_vec, a_vec, pos_vec - neg_vec

    def _forward(self, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for layer, module in self.modules.items():
            outputs[layer] = module(vec)
        return outputs

    def train_on_batch(
        self,
        samples: Sequence[TexGradSample],
        target_rank: int,
    ) -> Tuple[Dict[str, np.ndarray], TexGradSample, int]:
        """
        Perform a single optimization step and return LoRA deltas per layer.
        """
        if not samples:
            return {}, TexGradSample.blank(), 0

        params = self._lora_parameters()
        self._zero_lora_grads()

        lm_total = torch.zeros(1, device=self.device)
        ent_total = torch.zeros(1, device=self.device)
        cit_total = torch.zeros(1, device=self.device)
        ctr_total = torch.zeros(1, device=self.device)

        for sample in samples:
            q_vec, a_vec, contrast_vec = self._encode_sample(sample)
            # Forward pass through synthetic LoRA stack
            layer_outputs = self._forward(q_vec)
            primary = layer_outputs[next(iter(layer_outputs))]

            # Compute differentiable proxy losses
            lm_loss = 1 - torch.cosine_similarity(primary, a_vec.unsqueeze(0)).mean()

            ent_target = torch.tensor(
                self.entail_scorer.score(sample.answer, sample.positive_contexts),
                device=self.device,
            )
            ent_pred = torch.sigmoid((primary * a_vec.unsqueeze(0)).sum(dim=-1)).mean()
            entailment_loss = F.mse_loss(ent_pred, ent_target)

            cit_target = torch.tensor(
                self.citation_aligner.coverage(sample.answer, sample.positive_contexts),
                device=self.device,
            )
            dot_scores = torch.matmul(primary, contrast_vec.unsqueeze(0).t()).squeeze(1)
            coverage_pred = torch.softmax(dot_scores, dim=-1).max()
            citation_loss = F.mse_loss(coverage_pred, cit_target)

            contrastive_loss = torch.relu(torch.cosine_similarity(primary, contrast_vec.unsqueeze(0))).mean()

            lm_total = lm_total + lm_loss
            ent_total = ent_total + entailment_loss
            cit_total = cit_total + citation_loss
            ctr_total = ctr_total + contrastive_loss

        count = len(samples)
        lm_avg = lm_total / count
        ent_avg = ent_total / count
        cit_avg = cit_total / count
        ctr_avg = ctr_total / count

        self._zero_lora_grads()
        lm_avg.backward(retain_graph=True)
        lm_grads = self._capture_grads(params)

        self._zero_lora_grads()
        ent_avg.backward(retain_graph=True)
        entailment_grads = self._capture_grads(params)

        self._zero_lora_grads()
        cit_avg.backward(retain_graph=True)
        citation_grads = self._capture_grads(params)

        self._zero_lora_grads()
        ctr_avg.backward()
        contrastive_grads = self._capture_grads(params)

        lambdas = [
            self.texgrad_config.lambdas.get("ent", 0.5),
            self.texgrad_config.lambdas.get("attr", 0.5),
            self.texgrad_config.lambdas.get("ctr", 0.3),
        ]
        steered = project_gradient(lm_grads, [entailment_grads, citation_grads, contrastive_grads], lambdas)
        self._assign_grads(params, steered)

        self.optimizer.step()
        self._zero_lora_grads()

        layer_deltas: Dict[str, np.ndarray] = {}
        for layer, module in self.modules.items():
            delta = module.lora_delta().cpu().numpy()
            # Adjust rank via truncated SVD if target rank smaller than module rank
            if target_rank < module.rank:
                u, s, vh = np.linalg.svd(delta, full_matrices=False)
                delta = (u[:, :target_rank] * s[:target_rank]) @ vh[:target_rank, :]
            layer_deltas[layer] = delta.astype(np.float64)

        representative = self.texgrad_head.aggregate_metrics(samples)
        return layer_deltas, representative, len(samples)

    def _lora_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for layer in sorted(self.modules.keys()):
            module = self.modules[layer]
            params.append(module.lora_A)
            params.append(module.lora_B)
        return params

    def _zero_lora_grads(self) -> None:
        for param in self._lora_parameters():
            if param.grad is not None:
                param.grad.zero_()

    @staticmethod
    def _capture_grads(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
        grads: List[torch.Tensor] = []
        for param in params:
            if param.grad is None:
                grads.append(torch.zeros_like(param))
            else:
                grads.append(param.grad.detach().clone())
        return grads

    @staticmethod
    def _assign_grads(params: Sequence[torch.nn.Parameter], grads: Sequence[torch.Tensor | None]) -> None:
        for param, grad in zip(params, grads):
            if grad is None:
                param.grad = None
            else:
                if param.grad is None:
                    param.grad = grad.clone()
                else:
                    param.grad.copy_(grad)
