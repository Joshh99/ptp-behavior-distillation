"""
Ensemble: Run multiple strategies and combine results via fusion.

This is a meta-pattern that composes any of the other agentic patterns.
Unlike Orchestrator (which routes to ONE agent), Ensemble runs ALL strategies
and fuses their results.

Fusion strategies:
- MAJORITY_VOTE: Most common result wins
- WEIGHTED_AVERAGE: Weight by strategy weights
- LLM_JUDGE: LLM picks the best result
- FIRST_SUCCESS: Return first strategy that succeeds
- CUSTOM: User-provided fusion function

Example:
    >>> from ptool_framework.ensemble import Ensemble, FusionStrategy
    >>> ens = Ensemble(fusion=FusionStrategy.MAJORITY_VOTE)
    >>> ens.add_strategy("fast", my_fast_fn)
    >>> ens.add_strategy("careful", my_careful_fn)
    >>> result = ens.run("Classify this text")
    >>> print(result.winning_strategy)
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .llm_backend import call_llm, LLMResponse


# ============================================================================
# Enums
# ============================================================================

class FusionStrategy(Enum):
    """Strategy for fusing results from multiple strategies."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    LLM_JUDGE = "llm_judge"
    FIRST_SUCCESS = "first_success"
    CUSTOM = "custom"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class StrategyResult:
    """Result from one strategy in the ensemble."""
    strategy_name: str
    result: Optional[str] = None
    success: bool = True
    confidence: float = 1.0
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "result": self.result,
            "success": self.success,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyResult:
        return cls(
            strategy_name=data["strategy_name"],
            result=data.get("result"),
            success=data.get("success", True),
            confidence=data.get("confidence", 1.0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            error=data.get("error"),
        )


@dataclass
class EnsembleResult:
    """Fused output from ensemble of strategies."""
    final_result: Optional[str] = None
    strategy_results: List[StrategyResult] = field(default_factory=list)
    agreement_score: float = 0.0
    winning_strategy: Optional[str] = None
    fusion_used: FusionStrategy = FusionStrategy.MAJORITY_VOTE
    all_failed: bool = False

    @property
    def n_strategies(self) -> int:
        return len(self.strategy_results)

    @property
    def successful_results(self) -> List[StrategyResult]:
        return [r for r in self.strategy_results if r.success]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_result": self.final_result,
            "strategy_results": [r.to_dict() for r in self.strategy_results],
            "agreement_score": self.agreement_score,
            "winning_strategy": self.winning_strategy,
            "fusion_used": self.fusion_used.value,
            "all_failed": self.all_failed,
            "n_strategies": self.n_strategies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnsembleResult:
        return cls(
            final_result=data.get("final_result"),
            strategy_results=[StrategyResult.from_dict(r) for r in data.get("strategy_results", [])],
            agreement_score=data.get("agreement_score", 0.0),
            winning_strategy=data.get("winning_strategy"),
            fusion_used=FusionStrategy(data.get("fusion_used", "majority_vote")),
            all_failed=data.get("all_failed", False),
        )


# ============================================================================
# Ensemble Agent
# ============================================================================

class Ensemble:
    """
    Run multiple strategies and fuse their results.

    Args:
        fusion: How to combine results
        model: LLM model (for LLM_JUDGE fusion)
        llm_backend: Custom LLM backend (for testing)
        echo: Whether to print progress
        custom_fusion: User-provided fusion function
            Signature: (results: List[StrategyResult]) -> str
    """

    def __init__(
        self,
        fusion: FusionStrategy = FusionStrategy.MAJORITY_VOTE,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
        custom_fusion: Optional[Callable[[List[StrategyResult]], str]] = None,
    ):
        self.fusion = fusion
        self.model = model
        self.llm_backend = llm_backend
        self.echo = echo
        self.custom_fusion = custom_fusion

        if fusion == FusionStrategy.CUSTOM and custom_fusion is None:
            raise ValueError("CUSTOM fusion requires custom_fusion function")

        self._strategies: List[dict] = []  # [{name, fn, weight}]

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via backend or default call_llm."""
        if self.llm_backend:
            result = self.llm_backend(prompt, self.model)
            return result.content if isinstance(result, LLMResponse) else result
        return call_llm(prompt, self.model).content

    def add_strategy(
        self,
        name: str,
        fn: Callable[[str], str],
        weight: float = 1.0,
    ) -> "Ensemble":
        """
        Add a strategy to the ensemble.

        Args:
            name: Strategy name
            fn: Callable that takes a goal and returns a result string
            weight: Weight for weighted fusion

        Returns:
            self (for fluent API chaining)
        """
        self._strategies.append({
            "name": name,
            "fn": fn,
            "weight": weight,
        })
        return self

    def run(self, goal: str) -> EnsembleResult:
        """
        Run all strategies and fuse results.

        Args:
            goal: The task to accomplish

        Returns:
            EnsembleResult with fused output
        """
        if not self._strategies:
            return EnsembleResult(all_failed=True)

        # Execute all strategies
        strategy_results = []
        for strategy in self._strategies:
            if self.echo:
                print(f"  [Ensemble] Running strategy: {strategy['name']}...")

            start = time.time()
            try:
                result = strategy["fn"](goal)
                elapsed = (time.time() - start) * 1000
                strategy_results.append(StrategyResult(
                    strategy_name=strategy["name"],
                    result=str(result),
                    success=True,
                    confidence=strategy["weight"],
                    execution_time_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                strategy_results.append(StrategyResult(
                    strategy_name=strategy["name"],
                    result=None,
                    success=False,
                    execution_time_ms=elapsed,
                    error=str(e),
                ))

        # Check if all failed
        successful = [r for r in strategy_results if r.success]
        if not successful:
            return EnsembleResult(
                strategy_results=strategy_results,
                all_failed=True,
                fusion_used=self.fusion,
            )

        # Fuse results
        return self._fuse(strategy_results, successful)

    def _fuse(
        self,
        all_results: List[StrategyResult],
        successful: List[StrategyResult],
    ) -> EnsembleResult:
        """Fuse strategy results using the configured fusion strategy."""
        if self.fusion == FusionStrategy.MAJORITY_VOTE:
            return self._majority_vote(all_results, successful)
        elif self.fusion == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average(all_results, successful)
        elif self.fusion == FusionStrategy.LLM_JUDGE:
            return self._llm_judge(all_results, successful)
        elif self.fusion == FusionStrategy.FIRST_SUCCESS:
            return self._first_success(all_results, successful)
        elif self.fusion == FusionStrategy.CUSTOM:
            return self._custom_fuse(all_results, successful)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

    def _majority_vote(
        self,
        all_results: List[StrategyResult],
        successful: List[StrategyResult],
    ) -> EnsembleResult:
        """Majority vote across successful strategies."""
        votes = Counter(r.result for r in successful)
        winner, winner_count = votes.most_common(1)[0]
        total = len(successful)

        # Find which strategy produced the winner first
        winning_strategy = None
        for r in successful:
            if r.result == winner:
                winning_strategy = r.strategy_name
                break

        return EnsembleResult(
            final_result=winner,
            strategy_results=all_results,
            agreement_score=winner_count / total,
            winning_strategy=winning_strategy,
            fusion_used=FusionStrategy.MAJORITY_VOTE,
        )

    def _weighted_average(
        self,
        all_results: List[StrategyResult],
        successful: List[StrategyResult],
    ) -> EnsembleResult:
        """Weighted vote using strategy weights/confidence."""
        weighted: Dict[str, float] = {}
        for r in successful:
            weighted[r.result] = weighted.get(r.result, 0.0) + r.confidence

        winner = max(weighted, key=weighted.get)
        total_weight = sum(weighted.values())

        winning_strategy = None
        for r in successful:
            if r.result == winner:
                winning_strategy = r.strategy_name
                break

        votes = Counter(r.result for r in successful)

        return EnsembleResult(
            final_result=winner,
            strategy_results=all_results,
            agreement_score=weighted[winner] / total_weight if total_weight > 0 else 0.0,
            winning_strategy=winning_strategy,
            fusion_used=FusionStrategy.WEIGHTED_AVERAGE,
        )

    def _llm_judge(
        self,
        all_results: List[StrategyResult],
        successful: List[StrategyResult],
    ) -> EnsembleResult:
        """Use LLM to judge which strategy produced the best result."""
        candidate_list = "\n".join(
            f"Strategy '{r.strategy_name}': {r.result}"
            for r in successful
        )
        prompt = (
            f"You are evaluating results from multiple strategies.\n\n"
            f"{candidate_list}\n\n"
            f"Which strategy produced the best result? "
            f'Respond with ONLY a JSON object: {{"winner": "strategy_name"}}'
        )

        if self.echo:
            print("  [Ensemble] Running LLM judge for fusion...")
        response = self._call_llm(prompt)

        # Parse winner
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                winner_name = data.get("winner", successful[0].strategy_name)
            else:
                winner_name = successful[0].strategy_name
        except (json.JSONDecodeError, ValueError):
            winner_name = successful[0].strategy_name

        # Find the winning result
        winner_result = None
        for r in successful:
            if r.strategy_name == winner_name:
                winner_result = r.result
                break
        if winner_result is None:
            winner_name = successful[0].strategy_name
            winner_result = successful[0].result

        votes = Counter(r.result for r in successful)

        return EnsembleResult(
            final_result=winner_result,
            strategy_results=all_results,
            agreement_score=votes.get(winner_result, 1) / len(successful),
            winning_strategy=winner_name,
            fusion_used=FusionStrategy.LLM_JUDGE,
        )

    def _first_success(
        self,
        all_results: List[StrategyResult],
        successful: List[StrategyResult],
    ) -> EnsembleResult:
        """Return first successful strategy's result."""
        winner = successful[0]

        return EnsembleResult(
            final_result=winner.result,
            strategy_results=all_results,
            agreement_score=1.0,
            winning_strategy=winner.strategy_name,
            fusion_used=FusionStrategy.FIRST_SUCCESS,
        )

    def _custom_fuse(
        self,
        all_results: List[StrategyResult],
        successful: List[StrategyResult],
    ) -> EnsembleResult:
        """Use user-provided fusion function."""
        winner = self.custom_fusion(successful)

        winning_strategy = None
        for r in successful:
            if r.result == winner:
                winning_strategy = r.strategy_name
                break

        votes = Counter(r.result for r in successful)

        return EnsembleResult(
            final_result=winner,
            strategy_results=all_results,
            agreement_score=votes.get(winner, 1) / len(successful),
            winning_strategy=winning_strategy,
            fusion_used=FusionStrategy.CUSTOM,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def ensemble_run(
    goal: str,
    strategies: List[dict],
    fusion: FusionStrategy = FusionStrategy.MAJORITY_VOTE,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
    echo: bool = False,
) -> EnsembleResult:
    """
    Quick ensemble run.

    Args:
        goal: Task to accomplish
        strategies: List of {"name": str, "fn": callable, "weight": float}
        fusion: How to combine results
        model: LLM model (for LLM_JUDGE)
        llm_backend: Custom backend (for testing)
        echo: Print progress

    Returns:
        EnsembleResult with fused output
    """
    ens = Ensemble(fusion=fusion, model=model, llm_backend=llm_backend, echo=echo)
    for s in strategies:
        ens.add_strategy(s["name"], s["fn"], s.get("weight", 1.0))
    return ens.run(goal)
