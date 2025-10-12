"""
Retrieval utilities for SYNAPSE clients.

Modules here provide dynamic context selection that combines vector search
with symbolic queries over the shared compendium.
"""

from .context import RetrievalPlanner, RetrievalConfig
from .vector_store import HashedVectorStore

__all__ = ["RetrievalPlanner", "RetrievalConfig", "HashedVectorStore"]
