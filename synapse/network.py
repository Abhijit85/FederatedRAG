from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Optional

from synapse.knowledge.compendium import KnowledgePackage
from synapse.privacy.encryption import SynapseEncryptor


@dataclass
class NetworkLatency:
    client_to_edge: tuple = (0.01, 0.05)
    edge_to_server: tuple = (0.01, 0.05)


class NetworkSimulator:
    """
    Simulates asynchronous network transport with optional encryption.
    """

    def __init__(
        self,
        latency: Optional[NetworkLatency] = None,
        secret: Optional[str] = None,
    ) -> None:
        self.latency = latency or NetworkLatency()
        self.encryptor = SynapseEncryptor(secret or "synapse-default-secret")
        self.transmission_log = []

    async def _sleep_between(self, bounds: tuple) -> None:
        lower, upper = bounds
        await asyncio.sleep(random.uniform(lower, upper))

    async def transmit_client_to_edge(self, package: KnowledgePackage) -> KnowledgePackage:
        await self._sleep_between(self.latency.client_to_edge)
        self.transmission_log.append(("client->edge", package.source_id, len(package.artifacts)))
        # Package already encrypted by client privacy policy; return as-is.
        return package

    async def transmit_edge_to_server(self, package: KnowledgePackage) -> KnowledgePackage:
        await self._sleep_between(self.latency.edge_to_server)
        self.transmission_log.append(("edge->server", package.source_id, len(package.artifacts)))
        return package
