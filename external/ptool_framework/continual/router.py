"""
Domain-agnostic confidence router for layer selection.

Routes requests between Python layer (L2) and LLM Pipeline (L4)
based on confidence scores and learned patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .base import LayerResult, BackoffReason
from .config import ContinualConfig


@dataclass
class RouteDecision:
    """Decision about which layer to use."""
    use_python: bool                         # True = L2 Python, False = L4 Pipeline
    confidence: float                        # Confidence in this decision
    reason: str                              # Human-readable explanation
    entity_hint: Optional[str] = None        # Identified entity (e.g., calculator)


class ConfidenceRouter(ABC):
    """
    Abstract router for deciding between Python (L2) and Pipeline (L4).

    Routing logic:
    1. Try Python layer first
    2. If Python succeeds with high confidence, use that result
    3. If Python fails or has low confidence, back off to Pipeline
    4. Learn from successful Pipeline results to improve Python

    Subclasses implement domain-specific routing logic.
    """

    def __init__(
        self,
        config: ContinualConfig,
        python_layer: Callable,
        pipeline_layer: Callable,
    ):
        """
        Initialize router.

        Args:
            config: Continual learning configuration
            python_layer: Function to try Python calculation
            pipeline_layer: Function to run LLM pipeline
        """
        self.config = config
        self.python_layer = python_layer
        self.pipeline_layer = pipeline_layer

    @abstractmethod
    def should_backoff(self, python_result: LayerResult) -> bool:
        """
        Decide if we should back off from Python to Pipeline.

        Args:
            python_result: Result from Python layer attempt

        Returns:
            True if we should back off to Pipeline
        """
        pass

    @abstractmethod
    def route(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> LayerResult:
        """
        Route a request through the appropriate layer.

        Args:
            question: The question/task to solve
            context: Context (e.g., patient note)
            **kwargs: Domain-specific arguments

        Returns:
            LayerResult with computation result or backoff info
        """
        pass

    def _default_routing_logic(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> LayerResult:
        """
        Default routing implementation.

        Can be used by subclasses as a starting point.
        """
        # Step 1: Try Python layer
        python_result = self.python_layer(question, context, **kwargs)

        # Step 2: Check if we should back off
        if python_result.success and not self.should_backoff(python_result):
            return python_result

        # Step 3: Back off to Pipeline
        pipeline_result = self.pipeline_layer(
            question,
            context,
            entity_hint=python_result.identified_entity,
            **kwargs
        )

        # Merge token counts
        pipeline_result.input_tokens += python_result.input_tokens
        pipeline_result.output_tokens += python_result.output_tokens

        return pipeline_result
