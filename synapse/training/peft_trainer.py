from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np

from synapse.training.dp import DPConfig
from synapse.training.gradient import project_gradient
from synapse.training.lora import LoRALayerConfig
from synapse.training.texgrad import TexGradHead, TexGradConfig, TexGradSample
from synapse.training.texgrad_models import CitationAligner, EntailmentScorer

try:  # Optional heavy dependencies
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from opacus.grad_sample import GradSampleModule
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    prepare_model_for_kbit_training = None  # type: ignore
    GradSampleModule = None  # type: ignore


def _ensure_dependencies() -> None:
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None or get_peft_model is None:
        raise RuntimeError(
            "PEFTTexGradTrainer requires transformers, peft, and torch. "
            "Install them (e.g., `pip install torch transformers peft accelerate bitsandbytes`)."
        )
    if GradSampleModule is None:
        raise RuntimeError(
            "PEFTTexGradTrainer requires opacus (GradSampleModule) for DP accounting. "
            "Install it via `pip install opacus`."
        )


def _average_hidden_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _cosine_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    vec_a = vec_a / vec_a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vec_b = vec_b / vec_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return 1 - (vec_a * vec_b).sum(dim=-1)


@dataclass
class PEFTTexGradTrainer:
    """
    TexGrad trainer backed by a PEFT-wrapped causal LLM. Designed for environments
    where transformers/peft/torch are available. The base model remains frozen;
    only LoRA adapters receive updates.
    """

    model_id: str
    quantization: str
    lora_config: LoRALayerConfig
    texgrad_config: TexGradConfig = TexGradConfig()
    dp_config: DPConfig | None = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        _ensure_dependencies()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dp_config = self.dp_config or DPConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_kwargs = {"device_map": "auto"}
        quant_mode = (self.quantization or "").lower()
        if quant_mode in {"4bit", "qlora"}:
            quant_kwargs.update({"load_in_4bit": True})
        elif quant_mode in {"8bit"}:
            quant_kwargs.update({"load_in_8bit": True})

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **quant_kwargs,
        )
        base_model.config.use_cache = False

        if quant_mode in {"4bit", "qlora", "8bit"} and prepare_model_for_kbit_training is not None:
            base_model = prepare_model_for_kbit_training(base_model)

        target_modules = list(self.lora_config.resolve_layers())
        rank = max(self.lora_config.rank_choices)
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        self.base_model = get_peft_model(base_model, lora_cfg)
        self.base_model.print_trainable_parameters()
        self.base_model.to(self.device)

        self.model = GradSampleModule(self.base_model)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=1e-4)
        self.texgrad_head = TexGradHead(self.texgrad_config)
        self.entail_scorer = EntailmentScorer(device=self.device)
        self.citation_aligner = CitationAligner(device=self.device)

    def _build_prompt(self, sample: TexGradSample) -> Tuple[str, str]:
        context_lines = []
        for idx, text in enumerate(sample.positive_contexts[:3]):
            context_lines.append(f"Source[{idx+1}]: {text}")
        prompt = (
            "You are a factual assistant. Use the provided sources to answer faithfully.\n"
            f"Question: {sample.question}\n"
            + "\n".join(context_lines)
            + "\nAnswer:"
        )
        return prompt, sample.answer or ""

    def _compute_losses(
        self,
        outputs,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens: int,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        sample: TexGradSample,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lm_loss = outputs.loss

        answer_states = hidden_states[:, prompt_tokens:, :]
        if answer_states.numel() == 0:
            answer_states = hidden_states
        answer_summary = answer_states.mean(dim=1)

        positive_summary = positives.mean(dim=0, keepdim=True)
        entail_target = torch.tensor(
            self.entail_scorer.score(sample.answer, sample.positive_contexts),
            device=self.device,
        )
        entail_logits = torch.matmul(answer_summary, positive_summary.t()).squeeze(1)
        entail_pred = torch.sigmoid(entail_logits).mean()
        entail_loss = F.mse_loss(entail_pred, entail_target)

        coverage_scores = torch.matmul(answer_states, positives.t())
        coverage_probs = torch.softmax(coverage_scores, dim=-1)
        citation_loss = torch.relu(1 - coverage_probs.max(dim=-1).values).mean()

        cov_target = torch.tensor(
            self.citation_aligner.coverage(sample.answer, sample.positive_contexts),
            device=self.device,
        )
        if negatives.numel() > 0:
            negative_summary = negatives.mean(dim=0, keepdim=True)
            contrast_logits = torch.matmul(answer_summary, negative_summary.t())
            contrastive_loss = F.softplus(contrast_logits).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
        cit_logits = torch.matmul(answer_states, positives.t())
        cit_pred = torch.softmax(cit_logits, dim=-1).max(dim=-1).values.mean()
        citation_loss = F.mse_loss(cit_pred, cov_target)

        return lm_loss, entail_loss, citation_loss, contrastive_loss

    def _apply_dp(self) -> None:
        clip = float(self.dp_config.clip_norm)
        noise_multiplier = float(self.dp_config.noise_multiplier)
        if clip <= 0:
            return

        if GradSampleModule is not None and isinstance(self.model, GradSampleModule):
            per_sample_squares = None
            grad_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for param in self.model.parameters():
                grad_sample = getattr(param, "grad_sample", None)
                if grad_sample is None:
                    continue
                flat = grad_sample.reshape(grad_sample.shape[0], -1)
                squares = flat.pow(2).sum(dim=1)
                per_sample_squares = squares if per_sample_squares is None else per_sample_squares + squares
                grad_samples.append((param, grad_sample))

            if per_sample_squares is None:
                return
            norms = per_sample_squares.clamp(min=1e-12).sqrt()
            factors = (clip / norms).clamp(max=1.0)

            for param, grad_sample in grad_samples:
                reshape = factors.view(-1, *([1] * (grad_sample.dim() - 1)))
                clipped = (grad_sample * reshape).mean(dim=0)
                noise_std = noise_multiplier * clip
                if noise_std > 0:
                    clipped = clipped + torch.normal(
                        mean=0.0,
                        std=noise_std,
                        size=clipped.shape,
                        device=clipped.device,
                    )
                param.grad = clipped
                delattr(param, "grad_sample")
        else:
            # Fallback to global clipping + noise if grad-sample is unavailable.
            total_norm = 0.0
            grads = []
            for param in self.base_model.parameters():
                if param.grad is None:
                    continue
                grads.append(param.grad)
                total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = math.sqrt(total_norm)
            scale = min(1.0, clip / (total_norm + 1e-6))
            for grad in grads:
                grad.data.mul_(scale)
                noise_std = noise_multiplier * clip
                if noise_std > 0:
                    grad.data.add_(torch.normal(0, noise_std, size=grad.shape, device=grad.device))

    def _encode_contexts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(1, self.base_model.config.hidden_size, device=self.device)
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.base_model.base_model(**enc, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        return _average_hidden_state(hidden_states, enc["attention_mask"])

    def train_on_batch(
        self,
        samples: Sequence[TexGradSample],
        target_rank: int,
    ) -> Tuple[Dict[str, np.ndarray], TexGradSample, int]:
        if not samples:
            return {}, TexGradSample.blank(), 0

        self.model.train()
        params = self._lora_parameters()
        self.optimizer.zero_grad(set_to_none=True)
        self._clear_grad_samples(params)

        lm_total = torch.zeros(1, device=self.device)
        ent_total = torch.zeros(1, device=self.device)
        cit_total = torch.zeros(1, device=self.device)
        ctr_total = torch.zeros(1, device=self.device)

        for sample in samples:
            prompt, answer = self._build_prompt(sample)
            full_text = prompt + " " + answer
            tokenized = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024,
            ).to(self.device)
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.size(1)
            labels = tokenized["input_ids"].clone()
            labels[:, :prompt_tokens] = -100

            outputs = self.model(
                **tokenized,
                labels=labels,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]
            attention_mask = tokenized["attention_mask"]

            positives = self._encode_contexts(sample.positive_contexts).to(self.device)
            negatives = self._encode_contexts(sample.negative_contexts or [sample.question]).to(self.device)

            lm_loss, ent_loss, cit_loss, ctr_loss = self._compute_losses(
                outputs,
                hidden_states,
                attention_mask,
                prompt_tokens,
                positives,
                negatives,
                sample,
            )

            lm_total = lm_total + lm_loss
            ent_total = ent_total + ent_loss
            cit_total = cit_total + cit_loss
            ctr_total = ctr_total + ctr_loss

        count = len(samples)
        lm_avg = lm_total / count
        ent_avg = ent_total / count
        cit_avg = cit_total / count
        ctr_avg = ctr_total / count

        self.model.zero_grad(set_to_none=True)
        self._clear_grad_samples(params)
        lm_avg.backward(retain_graph=True)
        lm_grads, lm_grad_samples = self._capture_grads(params)

        self.model.zero_grad(set_to_none=True)
        self._clear_grad_samples(params)
        ent_avg.backward(retain_graph=True)
        ent_grads, ent_grad_samples = self._capture_grads(params)

        self.model.zero_grad(set_to_none=True)
        self._clear_grad_samples(params)
        cit_avg.backward(retain_graph=True)
        cit_grads, cit_grad_samples = self._capture_grads(params)

        self.model.zero_grad(set_to_none=True)
        self._clear_grad_samples(params)
        ctr_avg.backward()
        ctr_grads, ctr_grad_samples = self._capture_grads(params)

        lambdas = [
            self.texgrad_config.lambdas.get("ent", 0.5),
            self.texgrad_config.lambdas.get("attr", 0.5),
            self.texgrad_config.lambdas.get("ctr", 0.3),
        ]

        steered = project_gradient(lm_grads, [ent_grads, cit_grads, ctr_grads], lambdas)
        steered_samples = project_gradient(
            lm_grad_samples,
            [ent_grad_samples, cit_grad_samples, ctr_grad_samples],
            lambdas,
        )
        self._assign_grads(params, steered, steered_samples)

        self._apply_dp()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._clear_grad_samples(params)

        layer_deltas: Dict[str, np.ndarray] = {}
        for name, module in self.base_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_a = module.lora_A.weight.detach().cpu()
                lora_b = module.lora_B.weight.detach().cpu()
                delta = (lora_b @ lora_a).numpy()
                if target_rank < delta.shape[0] and target_rank < delta.shape[1]:
                    u, s, vh = np.linalg.svd(delta, full_matrices=False)
                    delta = (u[:, :target_rank] * s[:target_rank]) @ vh[:target_rank, :]
                layer_deltas[name] = delta

        metrics = self.texgrad_head.aggregate_metrics(samples)
        return layer_deltas, metrics, len(samples)

    def _lora_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for name, module in self.base_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                params.append(module.lora_A)
                params.append(module.lora_B)
        return params

    @staticmethod
    def _clear_grad_samples(params: Sequence[torch.nn.Parameter]) -> None:
        for param in params:
            if hasattr(param, "grad_sample"):
                delattr(param, "grad_sample")

    @staticmethod
    def _capture_grads(
        params: Sequence[torch.nn.Parameter],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor | None]]:
        grads: List[torch.Tensor] = []
        grad_samples: List[torch.Tensor | None] = []
        for param in params:
            if param.grad is None:
                grads.append(torch.zeros_like(param))
            else:
                grads.append(param.grad.detach().clone())

            grad_sample = getattr(param, "grad_sample", None)
            if grad_sample is None:
                grad_samples.append(None)
            else:
                grad_samples.append(grad_sample.detach().clone())
        return grads, grad_samples

    @staticmethod
    def _assign_grads(
        params: Sequence[torch.nn.Parameter],
        grads: Sequence[torch.Tensor | None],
        grad_samples: Sequence[torch.Tensor | None],
    ) -> None:
        for param, grad, grad_sample in zip(params, grads, grad_samples):
            if grad is None:
                param.grad = None
            else:
                if param.grad is None:
                    param.grad = grad.clone()
                else:
                    param.grad.copy_(grad)

            if grad_sample is None:
                if hasattr(param, "grad_sample"):
                    delattr(param, "grad_sample")
            else:
                param.grad_sample = grad_sample.clone()
