from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.knowledge.compendium import KnowledgeArtifact


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
    ) -> None:
        super().__init__(metadata, privacy_policy=privacy_policy)
        self.math_compendium_path = math_compendium_path
        self.math_training_path = math_training_path
        self.science_compendium_path = science_compendium_path
        self.science_dataset_path = science_dataset_path
        self.training_sample_limit = training_sample_limit

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
