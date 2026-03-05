"""
Domain-agnostic configuration for continual learning.

ContinualConfig provides base settings that apply across all domains.
Domain-specific configs (e.g., MedCalcContinualConfig) extend this base.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ContinualConfig:
    """
    Base configuration for continual learning - domain agnostic.

    Controls:
    - Whether learning is enabled
    - When to mine patterns
    - Confidence thresholds for routing
    - Pattern storage location
    """

    # Learning control
    learning_enabled: bool = True           # Disable for test evaluation
    mining_frequency: int = 10              # Mine patterns every N backoffs
    mining_model: str = "deepseek-v3-0324"  # LLM for pattern analysis

    # Confidence thresholds
    confidence_threshold: float = 0.7       # Use Python if confidence >= this
    min_pattern_confidence: float = 0.3     # Minimum confidence for pattern use

    # Decay settings
    decay_enabled: bool = True
    decay_rate: float = 0.05                # Confidence decay rate per day
    decay_interval_runs: int = 50           # Apply decay every N runs
    prune_threshold: float = 0.1            # Remove patterns below this

    # Storage
    pattern_store_path: str = "~/.ptool_patterns"

    # Export settings (for deploying learned patterns)
    export_format: str = "python"           # "python" or "json"
    export_path: Optional[str] = None

    def get_store_path(self) -> Path:
        """Get expanded pattern store path."""
        return Path(self.pattern_store_path).expanduser()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "learning_enabled": self.learning_enabled,
            "mining_frequency": self.mining_frequency,
            "mining_model": self.mining_model,
            "confidence_threshold": self.confidence_threshold,
            "min_pattern_confidence": self.min_pattern_confidence,
            "decay_enabled": self.decay_enabled,
            "decay_rate": self.decay_rate,
            "decay_interval_runs": self.decay_interval_runs,
            "prune_threshold": self.prune_threshold,
            "pattern_store_path": self.pattern_store_path,
            "export_format": self.export_format,
            "export_path": self.export_path,
        }
