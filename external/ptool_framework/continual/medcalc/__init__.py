"""
MedCalc-specific continual learning implementations.

Provides calculator-indexed pattern storage and learning
for medical calculation tasks.

Components:
- MedCalcContinualConfig: MedCalc-specific configuration
- MedCalcPatternStore: Calculator-indexed pattern storage
- MedCalcPatternMiner: Extraction and reasoning pattern mining
- PythonCalculatorLayer: Wraps calculators.py with backoff detection
- GuidedPipeline: L4 pipeline with learned pattern injection
- MedCalcContinualExperiment: Main experiment runner
"""

from .config import MedCalcContinualConfig
from .pattern_store import MedCalcPatternStore
from .pattern_miner import MedCalcPatternMiner
from .python_layer import PythonCalculatorLayer
from .guided_pipeline import GuidedPipeline
from .experiment import MedCalcContinualExperiment

__all__ = [
    "MedCalcContinualConfig",
    "MedCalcPatternStore",
    "MedCalcPatternMiner",
    "PythonCalculatorLayer",
    "GuidedPipeline",
    "MedCalcContinualExperiment",
]
