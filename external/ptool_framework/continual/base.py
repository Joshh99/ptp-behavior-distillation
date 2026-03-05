"""
Domain-agnostic base classes for continual learning.

Provides abstract base classes that can be extended for any domain:
- Pattern: A learned pattern with confidence tracking
- PatternStore: Abstract storage for patterns
- PatternMiner: Abstract pattern extraction from traces

These classes define the interface; domain-specific implementations
(e.g., MedCalcPatternStore) provide the actual behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid


# =============================================================================
# Enums
# =============================================================================

class BackoffReason(Enum):
    """Reasons why the Python layer backs off to LLM pipeline."""
    CALCULATOR_NOT_IDENTIFIED = "calculator_not_identified"
    RULE_BASED_CALCULATOR = "rule_based_calculator"
    EXTRACTION_FAILED = "extraction_failed"
    LOW_CONFIDENCE = "low_confidence"
    PATTERN_NOT_FOUND = "pattern_not_found"
    PYTHON_ERROR = "python_error"
    VALIDATION_FAILED = "validation_failed"
    OFFICIAL_VALIDATION_FAILED = "official_validation_failed"


class PatternStage(Enum):
    """Which pipeline stage a pattern applies to."""
    CALC_ID = "calc_id"          # Calculator identification
    EXTRACTION = "extraction"     # Value extraction
    REASONING = "reasoning"       # Medical/domain reasoning
    VALIDATION = "validation"     # Value validation


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Pattern:
    """
    A learned pattern from successful backoff handling.

    Patterns are indexed by:
    - stage: Which pipeline stage they apply to
    - entity: Domain-specific grouping (e.g., calculator name for MedCalc)

    Examples are stored as input/output pairs for few-shot prompting.
    """
    pattern_id: str
    stage: PatternStage
    entity: str                              # Domain-specific (e.g., calculator name)
    content: str                             # The pattern content (regex, keyword, etc.)
    confidence: float = 1.0                  # 0.0 to 1.0
    examples: List[Dict[str, Any]] = field(default_factory=list)  # [{"input": ..., "output": ...}]

    # Tracking
    times_used: int = 0
    times_helpful: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def helpfulness_rate(self) -> float:
        """Rate at which this pattern was helpful when used."""
        if self.times_used == 0:
            return 0.5  # Unknown
        return self.times_helpful / self.times_used

    @property
    def relevance_score(self) -> float:
        """Overall relevance score combining helpfulness and confidence."""
        return self.helpfulness_rate * self.confidence

    def reinforce(self, was_helpful: bool) -> None:
        """Update pattern based on usage outcome."""
        self.times_used += 1
        if was_helpful:
            self.times_helpful += 1
            self.confidence = min(1.0, self.confidence * 1.1)
        else:
            self.confidence = max(0.1, self.confidence * 0.9)
        self.last_used = datetime.now().isoformat()

    def apply_decay(self, days: int = 1, decay_rate: float = 0.05) -> None:
        """Apply time-based confidence decay."""
        self.confidence = max(0.1, self.confidence * (1 - decay_rate * days))

    def add_example(self, input_data: Any, output_data: Any) -> None:
        """Add an example input/output pair."""
        self.examples.append({
            "input": input_data,
            "output": output_data,
            "added_at": datetime.now().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "stage": self.stage.value,
            "entity": self.entity,
            "content": self.content,
            "confidence": self.confidence,
            "examples": self.examples,
            "times_used": self.times_used,
            "times_helpful": self.times_helpful,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            stage=PatternStage(data["stage"]),
            entity=data["entity"],
            content=data["content"],
            confidence=data.get("confidence", 1.0),
            examples=data.get("examples", []),
            times_used=data.get("times_used", 0),
            times_helpful=data.get("times_helpful", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create(
        cls,
        stage: PatternStage,
        entity: str,
        content: str,
        **kwargs
    ) -> "Pattern":
        """Factory method to create a new pattern with generated ID."""
        return cls(
            pattern_id=str(uuid.uuid4())[:8],
            stage=stage,
            entity=entity,
            content=content,
            **kwargs
        )


@dataclass
class LayerResult:
    """
    Result from attempting computation at a layer (L2 Python or L4 Pipeline).

    Contains either a success result or backoff information.
    """
    success: bool
    result: Optional[float] = None
    extracted_values: Optional[Dict[str, Any]] = None
    identified_entity: Optional[str] = None   # e.g., calculator_name
    method: str = "unknown"                   # "python" or "pipeline"

    # Backoff information (when success=False)
    backoff_reason: Optional[BackoffReason] = None
    backoff_details: Optional[str] = None

    # Token tracking for cost
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "extracted_values": self.extracted_values,
            "identified_entity": self.identified_entity,
            "method": self.method,
            "backoff_reason": self.backoff_reason.value if self.backoff_reason else None,
            "backoff_details": self.backoff_details,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class TraceRecord:
    """
    A record of a backoff trace for pattern mining.

    Collected when L2 backs off to L4 and L4 succeeds.
    """
    trace_id: str
    question: str
    context: str                              # e.g., patient_note
    entity: str                               # e.g., calculator_name
    extracted_values: Dict[str, Any]
    ground_truth: Optional[float] = None
    backoff_reason: Optional[BackoffReason] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "context": self.context,
            "entity": self.entity,
            "extracted_values": self.extracted_values,
            "ground_truth": self.ground_truth,
            "backoff_reason": self.backoff_reason.value if self.backoff_reason else None,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceRecord":
        """Create from dictionary."""
        backoff = None
        if data.get("backoff_reason"):
            backoff = BackoffReason(data["backoff_reason"])
        return cls(
            trace_id=data["trace_id"],
            question=data["question"],
            context=data["context"],
            entity=data["entity"],
            extracted_values=data["extracted_values"],
            ground_truth=data.get("ground_truth"),
            backoff_reason=backoff,
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Abstract Base Classes
# =============================================================================

class PatternStore(ABC):
    """
    Abstract base class for pattern storage.

    Subclasses implement domain-specific storage (e.g., by calculator for MedCalc).
    """

    @abstractmethod
    def get_patterns_for_entity(
        self,
        entity: str,
        stage: Optional[PatternStage] = None,
        min_confidence: float = 0.0,
    ) -> List[Pattern]:
        """
        Get patterns for a specific entity.

        Args:
            entity: Domain-specific grouping (e.g., calculator name)
            stage: Optional filter by pipeline stage
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching patterns sorted by relevance
        """
        pass

    @abstractmethod
    def get_patterns_for_stage(
        self,
        stage: PatternStage,
        limit: int = 10,
    ) -> List[Pattern]:
        """
        Get all patterns for a pipeline stage.

        Args:
            stage: Pipeline stage to filter by
            limit: Maximum patterns to return

        Returns:
            List of patterns sorted by relevance
        """
        pass

    @abstractmethod
    def store_pattern(self, pattern: Pattern) -> None:
        """Store a new pattern."""
        pass

    @abstractmethod
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing pattern."""
        pass

    @abstractmethod
    def reinforce_pattern(self, pattern_id: str, was_helpful: bool) -> None:
        """Reinforce or decay a pattern based on usage."""
        pass

    @abstractmethod
    def apply_decay(self, days: int = 1) -> int:
        """Apply decay to all patterns. Returns count affected."""
        pass

    @abstractmethod
    def prune_low_confidence(self, threshold: float = 0.1) -> int:
        """Remove patterns below confidence threshold. Returns count removed."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


class PatternMiner(ABC):
    """
    Abstract base class for pattern mining from traces.

    Subclasses implement domain-specific pattern extraction
    (e.g., regex patterns for MedCalc extraction).
    """

    @abstractmethod
    def add_trace(self, trace: TraceRecord) -> None:
        """Add a trace record for later mining."""
        pass

    @abstractmethod
    def mine_pending(self) -> List[Pattern]:
        """
        Mine patterns from pending traces.

        Returns newly discovered patterns.
        """
        pass

    @abstractmethod
    def get_pending_count(self) -> int:
        """Get count of traces pending mining."""
        pass

    @abstractmethod
    def clear_pending(self) -> None:
        """Clear pending traces."""
        pass
