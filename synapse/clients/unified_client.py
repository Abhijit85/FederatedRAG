from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.hyfical.contracts import (
    AdapterUpdate,
    LayerUpdate,
    PrivacyBudget,
    UpdateTelemetry,
)
from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.training import (
    DPConfig,
    DifferentialPrivacyGuard,
    LoRALayerConfig,
    LoRAUpdatePlanner,
    PEFTTexGradTrainer,
    SecAggAdapter,
    SecAggConfig,
    TexGradConfig,
    TexGradHead,
    TexGradSample,
    TexGradLoRATrainer,
)
from synapse.training.health import HealthAgent, HealthConfig
from synapse.utils import env_bool, env_int, env_str


class UnifiedQAClient(SynapseClient):
    """
    SYNAPSE client that exposes both MathQA and ScienceQA knowledge so each federated
    participant can contribute multi-domain artifacts.
    """

    def __init__(
        self,
        metadata: ClientMetadata,
        math_compendium_path: Path,
        math_training_path: Path,
        science_compendium_path: Path,
        science_dataset_path: Path,
        *,
        privacy_policy=None,
        training_sample_limit: int = 20,
        lora_config: LoRALayerConfig | None = None,
        texgrad_config: TexGradConfig | None = None,
        dp_config: DPConfig | None = None,
        secagg_config: SecAggConfig | None = None,
        health_config: HealthConfig | None = None,
    ) -> None:
        super().__init__(metadata, privacy_policy=privacy_policy)
        self.math_compendium_path = math_compendium_path
        self.math_training_path = math_training_path
        self.science_compendium_path = science_compendium_path
        self.science_dataset_path = science_dataset_path
        self.training_sample_limit = env_int("SYNAPSE_CLIENT_SAMPLE_LIMIT", training_sample_limit)

        client_lora_config = lora_config or LoRALayerConfig.from_env(prefix="SYNAPSE_CLIENT")
        dp_cfg = dp_config or DPConfig.from_env(prefix="SYNAPSE_CLIENT")
        secagg_cfg = secagg_config or SecAggConfig.from_env(prefix="SYNAPSE_CLIENT")
        health_cfg = health_config or HealthConfig.from_env(prefix="SYNAPSE_CLIENT")

        self.lora_planner = LoRAUpdatePlanner(client_lora_config)
        self.texgrad_head = TexGradHead(texgrad_config)
        self.dp_guard = DifferentialPrivacyGuard(dp_cfg)
        self.secagg = SecAggAdapter(secagg_cfg)
        self.health_agent = HealthAgent(health_cfg)
        rank_choices = self.lora_planner.config.rank_choices
        self._current_rank = max(rank_choices) if rank_choices else 8
        self._round_hint = 0
        # Track base model configuration via environment toggles.
        vlm_default = env_str("VLM_MODEL", "Llama-3-8B-Instruct") or "Llama-3-8B-Instruct"
        self.base_model_name = env_str("SYNAPSE_CLIENT_BASE_MODEL", vlm_default) or vlm_default
        self.quantization = env_str("SYNAPSE_CLIENT_QUANTIZATION", "4bit") or "4bit"
        self.use_peft = env_bool("SYNAPSE_CLIENT_USE_PEFT", False)
        default_quantized = self.quantization.lower() in {"4bit", "8bit", "qlora"}
        self.base_model_quantized = env_bool("SYNAPSE_CLIENT_BASE_MODEL_QUANTIZED", default_quantized)

        self.enable_backprop = env_bool("SYNAPSE_CLIENT_BACKPROP", True)
        self._trainer: TexGradLoRATrainer | PEFTTexGradTrainer | None = None

        if self.use_peft:
            try:
                self._trainer = PEFTTexGradTrainer(
                    model_id=self.base_model_name,
                    quantization=self.quantization,
                    lora_config=self.lora_planner.config,
                    dp_config=dp_cfg,
                )
            except RuntimeError:
                self._trainer = None

        if self._trainer is None and self.enable_backprop:
            try:
                self._trainer = TexGradLoRATrainer(self.lora_planner.config)
            except RuntimeError:
                self._trainer = None

    def _load_json_file(self, path: Path) -> Dict[str, object] | List[Dict[str, object]]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_math_compendium(self) -> Sequence[Dict[str, object]]:
        data = self._load_json_file(self.math_compendium_path)
        if isinstance(data, dict):
            scenarios = data.get("Usage_Scenarios")
            if scenarios:
                return scenarios
            textual = data.get("Textual_Compendium") or {}
            return textual.get("Usage_Scenarios", [])
        return []

    def _load_math_training(self) -> Sequence[Dict[str, object]]:
        data = self._load_json_file(self.math_training_path)
        if isinstance(data, dict):
            return data.get("examples", [])
        return data

    def _load_science_compendium(self) -> Sequence[Dict[str, object]]:
        data = self._load_json_file(self.science_compendium_path)
        if isinstance(data, dict):
            textual = data.get("Textual_Compendium") or {}
            return textual.get("Usage_Scenarios", [])
        return []

    def _load_science_dataset(self) -> Sequence[Dict[str, object]]:
        data = self._load_json_file(self.science_dataset_path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", [])
        return []

    def collect_local_artifacts(self):
        artifacts: List[KnowledgeArtifact] = []

        # Math usage scenarios
        for scenario in self._load_math_compendium():
            scenario_name = scenario.get("scenario") or scenario.get("name", "Unnamed Scenario")
            context = scenario.get("context", "")
            metadata = {
                "tool": "mathqa",
                "scenario": scenario_name,
                "pii_safe": True,
                "difficulty": scenario.get("difficulty", "medium"),
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"unified::math::scenario::{scenario_name}",
                    text=f"{scenario_name}: {context}",
                    structured_payload={
                        "type": "usage_scenario",
                        "skills": scenario.get("skills", []),
                        "example": scenario.get("example"),
                    },
                    metadata=metadata,
                )
            )

        # Math training examples
        for sample in self._load_math_training()[: self.training_sample_limit]:
            question = sample.get("Problem") or sample.get("question", "")
            solution = sample.get("Rationale") or sample.get("solution", "")
            options = sample.get("options")
            difficulty = sample.get("difficulty", "medium")
            structured = {
                "type": "training_example",
                "options": options,
                "answer": sample.get("Answer") or sample.get("correct"),
            }
            metadata = {
                "tool": "mathqa",
                "pii_safe": True,
                "difficulty": difficulty,
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"unified::math::example::{hash(question)}",
                    text=f"Problem: {question}\nSolution: {solution}",
                    structured_payload=structured,
                    metadata=metadata,
                )
            )

        science_privacy = {"pii_safe": True}

        # Science usage scenarios
        for scenario in self._load_science_compendium():
            scenario_name = scenario.get("scenario", "Unnamed Scenario")
            context = scenario.get("context", "")
            metadata = {
                **science_privacy,
                "tool": "scienceqa",
                "scenario": scenario_name,
                "domain": scenario.get("domain", "science"),
                "difficulty": scenario.get("difficulty", "medium"),
            }
            payload = {
                "type": "usage_scenario",
                "visual_skills": scenario.get("visual_skills", []),
                "textual_skills": scenario.get("skills", []),
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"unified::science::scenario::{scenario_name}",
                    text=f"{scenario_name}: {context}",
                    structured_payload=payload,
                    metadata=metadata,
                )
            )

        # Science training examples
        for sample in self._load_science_dataset()[: self.training_sample_limit]:
            question = sample.get("question", "")
            lecture = sample.get("lecture", "")
            choices = sample.get("choices", [])
            answer = sample.get("answer", "")
            has_image = bool(sample.get("image"))
            metadata = {
                **science_privacy,
                "tool": "scienceqa",
                "difficulty": sample.get("difficulty", "medium"),
                "topic": sample.get("topic", ""),
            }
            payload = {
                "type": "training_example",
                "choices": choices,
                "answer": answer,
                "has_image": has_image,
            }
            text_block = f"Question: {question}\nLecture: {lecture}"
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"unified::science::example::{hash(question)}",
                    text=text_block,
                    structured_payload=payload,
                    metadata=metadata,
                )
            )

        return artifacts

    def _gather_math_samples(self, limit: int) -> List[TexGradSample]:
        data = list(self._load_math_training()[:limit])
        samples: List[TexGradSample] = []
        if not data:
            return samples
        for idx, sample in enumerate(data):
            question = sample.get("Problem") or sample.get("question", "")
            answer = str(sample.get("Answer") or sample.get("correct") or "")
            rationale = sample.get("Rationale") or sample.get("solution", "")
            positives = [rationale]
            options = sample.get("options")
            if isinstance(options, list):
                positives.append(" ".join(options))
            negative_idx = (idx + 1) % len(data)
            negative_context = data[negative_idx].get("Rationale") or data[negative_idx].get("solution", "")
            negatives = [negative_context] if negative_context else []
            samples.append(
                TexGradSample.from_strings(
                    question=question,
                    answer=answer,
                    positives=positives,
                    negatives=negatives,
                )
            )
        return samples

    def _gather_science_samples(self, limit: int) -> List[TexGradSample]:
        data = list(self._load_science_dataset()[:limit])
        samples: List[TexGradSample] = []
        if not data:
            return samples

        for idx, sample in enumerate(data):
            question = sample.get("question", "")
            answer = str(sample.get("answer", ""))
            lecture = sample.get("lecture", "")
            positives = [lecture]
            caption = sample.get("image")
            if caption:
                positives.append(str(caption))
            negative_idx = (idx + 1) % len(data)
            negative_lecture = data[negative_idx].get("lecture", "")
            negatives = [negative_lecture] if negative_lecture else []
            samples.append(
                TexGradSample.from_strings(
                    question=question,
                    answer=answer,
                    positives=positives,
                    negatives=negatives,
                )
            )
        return samples

    def _gather_samples(self) -> List[TexGradSample]:
        math_samples = self._gather_math_samples(self.training_sample_limit)
        science_samples = self._gather_science_samples(self.training_sample_limit)
        return math_samples + science_samples

    def prepare_adapter_update(self) -> AdapterUpdate:
        """
        Produce a federated adapter update with TexGrad telemetry and DP noise.
        """
        self.health_agent.heartbeat()
        self.secagg.next_round()
        self._round_hint += 1

        samples = self._gather_samples()
        lora_rank = self._current_rank

        if self._trainer is not None:
            try:
                layer_vectors, metrics, steps = self._trainer.train_on_batch(samples, lora_rank)
                texgrad_metrics = metrics
                steps_count = steps
            except RuntimeError:
                layer_vectors = self.lora_planner.build_layer_updates(samples, rank=lora_rank)
                texgrad_metrics = self.texgrad_head.aggregate_metrics(samples)
                steps_count = len(samples)
        else:
            layer_vectors = self.lora_planner.build_layer_updates(samples, rank=lora_rank)
            texgrad_metrics = self.texgrad_head.aggregate_metrics(samples)
            steps_count = len(samples)

        layer_updates: List[LayerUpdate] = []
        for layer, vector in layer_vectors.items():
            sanitized = self.dp_guard.sanitize(vector)
            masked, metadata, norm = self.secagg.mask(
                self.metadata.client_id,
                self._round_hint,
                layer,
                sanitized,
            )
            layer_updates.append(
                LayerUpdate(
                    layer=layer,
                    format="LoRA",
                    rank=lora_rank,
                    delta_hash=self.lora_planner.delta_hash(sanitized),
                    masked_delta=masked,
                    norm=norm,
                    mask_metadata=metadata,
                )
            )

        steps = max(steps_count, 1)
        epsilon_local = self.dp_guard.estimate_local_epsilon(steps)
        telemetry = UpdateTelemetry(
            freshness_ts=int(datetime.now(timezone.utc).timestamp()),
            steps=steps,
            loss_lm=max(0.5, 2.0 - texgrad_metrics.entailment - texgrad_metrics.citation_coverage),
            texgrad=texgrad_metrics,
        )

        dp_budget = PrivacyBudget(
            clipping=self.dp_guard.clip_norm(),
            sigma=self.dp_guard.sigma(),
            epsilon_local=epsilon_local,
        )

        return AdapterUpdate(
            client_id=self.metadata.client_id,
            round_hint=self._round_hint,
            layer_updates=layer_updates,
            telemetry=telemetry,
            dp_local=dp_budget,
        )

    def apply_global_bundle(self, adapter_bundle) -> None:
        """
        Update local routing hints from a GlobalAdapterBundle instance.
        """
        if adapter_bundle is None:
            return
        self.health_agent.mark_ack(adapter_bundle.version)
        router_hint = adapter_bundle.router_hints or {}
        # Use hint to adjust next rank selection if available.
        for layer, experts in adapter_bundle.adapters.items():
            if not experts:
                continue
            preferred_rank = max(expert.rank for expert in experts)
            self._current_rank = preferred_rank
