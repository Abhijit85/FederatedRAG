"""
HyFICAL (Hybrid Federated Intelligence Control & Aggregation Layer).

This package provides scheduling, secure aggregation adapters, robust
aggregation, adapter routing, and compliance instrumentation aligned
with the enhanced SYNAPSE federation design.
"""

from .contracts import (
    AdapterUpdate,
    GlobalAdapterBundle,
    LayerUpdate,
    PrivacyBudget,
    TexGradMetrics,
    UpdateTelemetry,
)
from .scheduler import AsyncWindowScheduler, SchedulerConfig
from .router import AdapterRouter, RouterConfig
from .aggregator import HyFICALAggregator, AggregationConfig, AggregatedLayerResult

__all__ = [
    "AdapterUpdate",
    "GlobalAdapterBundle",
    "LayerUpdate",
    "PrivacyBudget",
    "TexGradMetrics",
    "UpdateTelemetry",
    "AsyncWindowScheduler",
    "SchedulerConfig",
    "AdapterRouter",
    "RouterConfig",
    "HyFICALAggregator",
    "AggregationConfig",
    "AggregatedLayerResult",
]
