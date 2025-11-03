"""
Client-side training utilities: LoRA adapters, TexGrad steering, DP guard,
Secure Aggregation helpers, and health monitoring.
"""

from .lora import LoRALayerConfig, LoRAUpdatePlanner
from .texgrad import TexGradConfig, TexGradHead, TexGradSample
from .dp import DifferentialPrivacyGuard, DPConfig
from .secagg import SecAggAdapter, SecAggConfig
from .health import HealthAgent, HealthConfig
from .backprop import TexGradLoRATrainer
from .peft_trainer import PEFTTexGradTrainer
from .gradient import project_gradient

__all__ = [
    "LoRALayerConfig",
    "LoRAUpdatePlanner",
    "TexGradConfig",
    "TexGradHead",
    "TexGradSample",
    "DifferentialPrivacyGuard",
    "DPConfig",
    "SecAggAdapter",
    "SecAggConfig",
    "HealthAgent",
    "HealthConfig",
    "TexGradLoRATrainer",
    "PEFTTexGradTrainer",
    "project_gradient",
]
