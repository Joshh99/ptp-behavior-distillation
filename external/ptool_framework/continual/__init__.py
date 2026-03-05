"""
L5 Continual Learning Framework.

Progressive learning from L2 (Python) to L4 (LLM Pipeline) backoff.

Core vision:
- Start cheap: Try L2 Python patterns first (fast, free)
- Back off when needed: Use L4 for rule-based scoring systems
- Learn from backoffs: Mine patterns from successful L4 extractions

Architecture:
    ptool_framework/continual/
        base.py         - Domain-agnostic: Pattern, PatternStore, PatternMiner ABCs
        config.py       - Domain-agnostic: ContinualConfig base class
        router.py       - Domain-agnostic: ConfidenceRouter base class
        medcalc/        - MedCalc-specific implementations

Example usage:
    from ptool_framework.continual.medcalc import MedCalcContinualExperiment

    experiment = MedCalcContinualExperiment(config)
    result = experiment.run_instance(instance)
"""

from .base import (
    Pattern,
    PatternStore,
    PatternMiner,
    BackoffReason,
    LayerResult,
)
from .config import ContinualConfig
from .router import ConfidenceRouter

__all__ = [
    # Base classes
    "Pattern",
    "PatternStore",
    "PatternMiner",
    "BackoffReason",
    "LayerResult",
    # Config
    "ContinualConfig",
    # Router
    "ConfidenceRouter",
]
