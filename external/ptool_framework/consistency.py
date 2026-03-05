"""
Self-Consistency: Sample N independent LLM responses, aggregate via majority vote.

Paper: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (ICLR 2023)

This module implements two patterns:
- SelfConsistency: Sample N responses, aggregate via majority vote, weighted vote,
  LLM judge, or custom function.
- BestOfN: Subclass that uses LLM_JUDGE by default (pick the best from N candidates).

Key insight: Python controls the sampling loop and aggregation logic.
The LLM only generates individual responses — all voting/judging is Python-orchestrated.

Example:
    >>> from ptool_framework.consistency import majority_vote
    >>> result = majority_vote("What is 6*7?", n=5, model="deepseek-v3-0324")
    >>> print(result.final_result)  # "42"
    >>> print(result.agreement_score)  # 1.0
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from .llm_backend import call_llm, execute_ptool, LLMResponse
from .ptool import PToolSpec
from .traces import WorkflowTrace, TraceStep, StepStatus


# ============================================================================
# Enums
# ============================================================================

class AggregationStrategy(Enum):
    """Strategy for aggregating N candidate responses."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    LLM_JUDGE = "llm_judge"
    CUSTOM = "custom"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CandidateResult:
    """One of N sampled LLM responses."""
    result: str
    index: int
    confidence: float = 1.0
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "index": self.index,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateResult:
        return cls(
            result=data["result"],
            index=data["index"],
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning"),
        )


@dataclass
class ConsistencyResult:
    """Aggregated output from Self-Consistency sampling."""
    final_result: str
    candidates: List[CandidateResult] = field(default_factory=list)
    agreement_score: float = 0.0
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_used: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE
    judge_reasoning: Optional[str] = None

    @property
    def n_samples(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_result": self.final_result,
            "candidates": [c.to_dict() for c in self.candidates],
            "agreement_score": self.agreement_score,
            "vote_distribution": self.vote_distribution,
            "strategy_used": self.strategy_used.value,
            "judge_reasoning": self.judge_reasoning,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConsistencyResult:
        return cls(
            final_result=data["final_result"],
            candidates=[CandidateResult.from_dict(c) for c in data.get("candidates", [])],
            agreement_score=data.get("agreement_score", 0.0),
            vote_distribution=data.get("vote_distribution", {}),
            strategy_used=AggregationStrategy(data.get("strategy_used", "majority_vote")),
            judge_reasoning=data.get("judge_reasoning"),
        )

    def to_workflow_trace(self) -> WorkflowTrace:
        """Convert to WorkflowTrace for standard trace storage."""
        trace = WorkflowTrace(goal=f"Self-Consistency ({self.n_samples} samples)")
        for candidate in self.candidates:
            step = TraceStep(
                ptool_name="self_consistency_sample",
                args={"index": candidate.index},
                goal=f"Sample {candidate.index + 1} of {self.n_samples}",
            )
            step.status = StepStatus.COMPLETED
            step.result = candidate.result
            trace.steps.append(step)

        # Add aggregation step
        agg_step = TraceStep(
            ptool_name="self_consistency_aggregate",
            args={
                "strategy": self.strategy_used.value,
                "vote_distribution": self.vote_distribution,
            },
            goal="Aggregate candidates",
        )
        agg_step.status = StepStatus.COMPLETED
        agg_step.result = self.final_result
        trace.steps.append(agg_step)
        return trace


# ============================================================================
# Self-Consistency Agent
# ============================================================================

class SelfConsistency:
    """
    Sample N independent LLM responses and aggregate them.

    Args:
        n: Number of samples to generate
        model: LLM model for generation
        strategy: How to aggregate candidates
        llm_backend: Custom LLM backend (for testing)
        echo: Whether to print progress
        custom_aggregator: User-provided aggregation function
            Signature: (candidates: List[CandidateResult]) -> str
    """

    def __init__(
        self,
        n: int = 5,
        model: str = "deepseek-v3-0324",
        strategy: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE,
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
        custom_aggregator: Optional[Callable[[List[CandidateResult]], str]] = None,
    ):
        self.n = n
        self.model = model
        self.strategy = strategy
        self.llm_backend = llm_backend
        self.echo = echo
        self.custom_aggregator = custom_aggregator

        if strategy == AggregationStrategy.CUSTOM and custom_aggregator is None:
            raise ValueError("CUSTOM strategy requires custom_aggregator function")

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via backend or default call_llm."""
        if self.llm_backend:
            result = self.llm_backend(prompt, self.model)
            return result.content if isinstance(result, LLMResponse) else result
        return call_llm(prompt, self.model).content

    def sample(self, prompt: str) -> ConsistencyResult:
        """
        Sample N responses for a prompt and aggregate.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            ConsistencyResult with final_result and all candidates
        """
        candidates = []

        for i in range(self.n):
            if self.echo:
                print(f"  [Self-Consistency] Sampling {i + 1}/{self.n}...")
            response = self._call_llm(prompt)
            # Extract the core answer — strip whitespace for consistency
            answer = response.strip()
            candidates.append(CandidateResult(
                result=answer,
                index=i,
            ))

        return self._aggregate(candidates)

    def sample_ptool(self, spec: PToolSpec, inputs: Dict[str, Any]) -> ConsistencyResult:
        """
        Sample N responses using a PToolSpec and aggregate.

        Args:
            spec: The ptool specification
            inputs: Dictionary of input arguments

        Returns:
            ConsistencyResult with final_result and all candidates
        """
        prompt = spec.format_prompt(**inputs)
        return self.sample(prompt)

    def _aggregate(self, candidates: List[CandidateResult]) -> ConsistencyResult:
        """Aggregate candidates using the configured strategy."""
        if self.strategy == AggregationStrategy.MAJORITY_VOTE:
            return self._majority_vote(candidates)
        elif self.strategy == AggregationStrategy.WEIGHTED_VOTE:
            return self._weighted_vote(candidates)
        elif self.strategy == AggregationStrategy.LLM_JUDGE:
            return self._llm_judge(candidates)
        elif self.strategy == AggregationStrategy.CUSTOM:
            return self._custom_aggregate(candidates)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _majority_vote(self, candidates: List[CandidateResult]) -> ConsistencyResult:
        """Simple majority vote — most common answer wins."""
        votes = Counter(c.result for c in candidates)
        winner, winner_count = votes.most_common(1)[0]
        total = len(candidates)

        return ConsistencyResult(
            final_result=winner,
            candidates=candidates,
            agreement_score=winner_count / total,
            vote_distribution=dict(votes),
            strategy_used=AggregationStrategy.MAJORITY_VOTE,
        )

    def _weighted_vote(self, candidates: List[CandidateResult]) -> ConsistencyResult:
        """Weighted vote using candidate confidence scores."""
        weighted_votes: Dict[str, float] = {}
        for c in candidates:
            weighted_votes[c.result] = weighted_votes.get(c.result, 0.0) + c.confidence

        winner = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())

        # Normalize for agreement score
        count_votes = Counter(c.result for c in candidates)

        return ConsistencyResult(
            final_result=winner,
            candidates=candidates,
            agreement_score=weighted_votes[winner] / total_weight if total_weight > 0 else 0.0,
            vote_distribution=dict(count_votes),
            strategy_used=AggregationStrategy.WEIGHTED_VOTE,
        )

    def _llm_judge(self, candidates: List[CandidateResult]) -> ConsistencyResult:
        """Use LLM to judge which candidate is best."""
        # Build judge prompt
        candidate_list = "\n".join(
            f"Candidate {i + 1}: {c.result}" for i, c in enumerate(candidates)
        )
        judge_prompt = (
            f"You are a judge evaluating {len(candidates)} candidate answers.\n\n"
            f"{candidate_list}\n\n"
            f"Which candidate has the best answer? Respond with ONLY a JSON object:\n"
            f'{{"winner": <number>, "reasoning": "<brief explanation>"}}'
        )

        if self.echo:
            print("  [Self-Consistency] Running LLM judge...")
        judge_response = self._call_llm(judge_prompt)

        # Parse judge response
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[^}]+\}', judge_response)
            if json_match:
                data = json.loads(json_match.group())
                winner_idx = int(data.get("winner", 1)) - 1  # 1-indexed to 0-indexed
                reasoning = data.get("reasoning", "")
            else:
                # Fallback: look for a number
                num_match = re.search(r'\d+', judge_response)
                winner_idx = int(num_match.group()) - 1 if num_match else 0
                reasoning = judge_response.strip()
        except (json.JSONDecodeError, ValueError, IndexError):
            winner_idx = 0
            reasoning = "Failed to parse judge response"

        # Clamp to valid range
        winner_idx = max(0, min(winner_idx, len(candidates) - 1))

        votes = Counter(c.result for c in candidates)
        winner = candidates[winner_idx].result

        return ConsistencyResult(
            final_result=winner,
            candidates=candidates,
            agreement_score=votes.get(winner, 1) / len(candidates),
            vote_distribution=dict(votes),
            strategy_used=AggregationStrategy.LLM_JUDGE,
            judge_reasoning=reasoning,
        )

    def _custom_aggregate(self, candidates: List[CandidateResult]) -> ConsistencyResult:
        """Use user-provided aggregation function."""
        winner = self.custom_aggregator(candidates)
        votes = Counter(c.result for c in candidates)

        return ConsistencyResult(
            final_result=winner,
            candidates=candidates,
            agreement_score=votes.get(winner, 1) / len(candidates),
            vote_distribution=dict(votes),
            strategy_used=AggregationStrategy.CUSTOM,
        )


# ============================================================================
# Best-of-N (subclass)
# ============================================================================

class BestOfN(SelfConsistency):
    """
    Best-of-N: Generate N candidates, use LLM judge to pick the best one.

    A specialization of SelfConsistency with LLM_JUDGE as the default strategy.
    Useful for open-ended generation where majority vote doesn't apply.

    Example:
        >>> best = BestOfN(n=3, model="deepseek-v3-0324")
        >>> result = best.sample("Write a haiku about coding")
        >>> print(result.final_result)  # Best haiku selected by judge
    """

    def __init__(
        self,
        n: int = 3,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
    ):
        super().__init__(
            n=n,
            model=model,
            strategy=AggregationStrategy.LLM_JUDGE,
            llm_backend=llm_backend,
            echo=echo,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def majority_vote(
    prompt: str,
    n: int = 5,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
    echo: bool = False,
) -> ConsistencyResult:
    """
    Quick majority vote over N LLM samples.

    Args:
        prompt: The prompt to send to the LLM
        n: Number of samples
        model: LLM model
        llm_backend: Custom backend (for testing)
        echo: Print progress

    Returns:
        ConsistencyResult with majority-selected answer
    """
    sc = SelfConsistency(
        n=n, model=model, strategy=AggregationStrategy.MAJORITY_VOTE,
        llm_backend=llm_backend, echo=echo,
    )
    return sc.sample(prompt)


def best_of_n(
    prompt: str,
    n: int = 3,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
    echo: bool = False,
) -> ConsistencyResult:
    """
    Quick best-of-N using LLM judge.

    Args:
        prompt: The prompt to send to the LLM
        n: Number of samples
        model: LLM model
        llm_backend: Custom backend (for testing)
        echo: Print progress

    Returns:
        ConsistencyResult with judge-selected best answer
    """
    bon = BestOfN(n=n, model=model, llm_backend=llm_backend, echo=echo)
    return bon.sample(prompt)
