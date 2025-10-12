from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.knowledge.compendium import KnowledgeArtifact


class ScienceQAClient(SynapseClient):
    """
    SYNAPSE client that extracts multimodal ScienceQA knowledge.
    """

    def __init__(
        self,
        metadata: ClientMetadata,
        compendium_path: Path,
        dataset_path: Path,
        privacy_policy=None,
    ) -> None:
        super().__init__(metadata, privacy_policy=privacy_policy)
        self.compendium_path = compendium_path
        self.dataset_path = dataset_path

    def _load_compendium(self) -> Dict[str, object]:
        with self.compendium_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_dataset(self) -> List[Dict[str, object]]:
        with self.dataset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else data.get("data", [])

    def _compute_image_embedding(self, image_b64: str) -> List[float]:
        try:
            if not image_b64:
                return []
            if "base64," in image_b64:
                image_b64 = image_b64.split("base64,")[1]
            image_bytes = base64.b64decode(image_b64)
            with Image.open(BytesIO(image_bytes)) as img:
                resized = img.convert("L").resize((16, 16))
                arr = np.array(resized, dtype=np.float32)
                arr /= 255.0
                histogram, _ = np.histogram(arr, bins=16, range=(0.0, 1.0))
                histogram = histogram.astype(np.float32)
                histogram /= np.linalg.norm(histogram) or 1.0
                features = np.concatenate([
                    histogram,
                    arr.mean(axis=0),
                    arr.mean(axis=1),
                    np.array([arr.mean(), arr.std()])
                ])
                return features.tolist()
        except Exception:
            return []
        return []

    def collect_local_artifacts(self):
        compendium = self._load_compendium()
        textual_section = compendium.get("Textual_Compendium", {})
        scenarios = textual_section.get("Usage_Scenarios", [])
        privacy_metadata = {"tool": "scienceqa", "pii_safe": True}

        artifacts: List[KnowledgeArtifact] = []
        for item in scenarios:
            scenario_name = item.get("scenario", "Unnamed Scenario")
            context = item.get("context", "")
            metadata = {
                **privacy_metadata,
                "scenario": scenario_name,
                "domain": item.get("domain", "science"),
                "difficulty": item.get("difficulty", "medium"),
            }
            structured_payload = {
                "type": "usage_scenario",
                "visual_skills": item.get("visual_skills", []),
                "textual_skills": item.get("skills", []),
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"scienceqa::scenario::{scenario_name}",
                    text=f"{scenario_name}: {context}",
                    structured_payload=structured_payload,
                    metadata=metadata,
                )
            )

        for sample in self._load_dataset()[:20]:
            question = sample.get("question", "")
            lecture = sample.get("lecture", "")
            choices = sample.get("choices", [])
            answer = sample.get("answer", "")
            has_image = bool(sample.get("image"))

            image_embedding = self._compute_image_embedding(sample.get("image", "")) if has_image else []

            structured_payload = {
                "type": "training_example",
                "choices": choices,
                "answer": answer,
                "has_image": has_image,
                "image_embedding": image_embedding,
            }
            metadata = {
                **privacy_metadata,
                "difficulty": sample.get("difficulty", "medium"),
                "topic": sample.get("topic", ""),
            }
            text_block = f"Question: {question}\nLecture: {lecture}"
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"scienceqa::example::{hash(question)}",
                    text=text_block,
                    structured_payload=structured_payload,
                    metadata=metadata,
                )
            )

        return artifacts
