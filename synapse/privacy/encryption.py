from __future__ import annotations

import base64
import hashlib
import os
from typing import Iterable, Tuple

from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage


def _derive_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


class SynapseEncryptor:
    """
    Lightweight XOR-based encryptor to protect artifacts in transit.

    NOTE: This is a pragmatic placeholder; production deployments should
    swap in a stronger cryptographic primitive.
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret or os.environ.get("SYNAPSE_SECRET", "synapse-default-secret")
        self._key = _derive_key(self._secret)

    def encrypt_text(self, text: str) -> str:
        data = text.encode("utf-8")
        cipher = _xor_bytes(data, self._key)
        return base64.b64encode(cipher).decode("utf-8")

    def decrypt_text(self, cipher_b64: str) -> str:
        data = base64.b64decode(cipher_b64.encode("utf-8"))
        plain = _xor_bytes(data, self._key)
        return plain.decode("utf-8")

    def encrypt_package(self, package: KnowledgePackage) -> KnowledgePackage:
        encrypted_artifacts = []
        for artifact in package.artifacts:
            encrypted_artifacts.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=self.encrypt_text(artifact.text),
                    structured_payload=artifact.structured_payload,
                    metadata={**artifact.metadata, "_encrypted": True},
                )
            )
        return KnowledgePackage(
            source_id=package.source_id,
            artifacts=encrypted_artifacts,
            created_at=package.created_at,
            metadata=package.metadata,
        )

    def decrypt_package(self, package: KnowledgePackage) -> KnowledgePackage:
        decrypted_artifacts = []
        for artifact in package.artifacts:
            is_encrypted = artifact.metadata.pop("_encrypted", False)
            text = self.decrypt_text(artifact.text) if is_encrypted else artifact.text
            decrypted_artifacts.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=text,
                    structured_payload=artifact.structured_payload,
                    metadata=artifact.metadata,
                )
            )
        return KnowledgePackage(
            source_id=package.source_id,
            artifacts=decrypted_artifacts,
            created_at=package.created_at,
            metadata=package.metadata,
        )
