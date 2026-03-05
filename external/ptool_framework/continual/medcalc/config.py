"""
MedCalc-specific configuration for continual learning.

Extends the base ContinualConfig with MedCalc-specific settings
like backoff behavior for rule-based calculators.
"""

from dataclasses import dataclass
from pathlib import Path

from ..config import ContinualConfig


@dataclass
class MedCalcContinualConfig(ContinualConfig):
    """
    MedCalc-specific continual learning configuration.

    Adds:
    - backoff_on_rule_based: Whether to always back off for scoring systems
    - Path defaults to medcalc subfolder
    """

    # MedCalc-specific backoff behavior
    backoff_on_rule_based: bool = True      # Always back off for scoring systems

    # Override default path to include medcalc subfolder
    pattern_store_path: str = "~/.ptool_patterns/medcalc"

    # Extraction confidence thresholds
    extraction_confidence_threshold: float = 0.6   # Minimum extraction confidence

    # Learning from L4 successes
    min_examples_for_pattern: int = 2       # Minimum examples before pattern is usable
    max_examples_per_pattern: int = 5       # Cap examples to avoid prompt bloat

    def get_store_path(self) -> Path:
        """Get expanded pattern store path."""
        return Path(self.pattern_store_path).expanduser()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        base = super().to_dict()
        base.update({
            "backoff_on_rule_based": self.backoff_on_rule_based,
            "extraction_confidence_threshold": self.extraction_confidence_threshold,
            "min_examples_for_pattern": self.min_examples_for_pattern,
            "max_examples_per_pattern": self.max_examples_per_pattern,
        })
        return base
