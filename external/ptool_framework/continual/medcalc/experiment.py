"""
MedCalc Continual Learning Experiment.

Main experiment class that orchestrates:
1. Python layer attempts (L2)
2. Backoff to guided pipeline (L4) when needed
3. Learning from successful backoffs
4. Metrics tracking

This is the core implementation of L5 Continual for MedCalc.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..base import LayerResult, BackoffReason, TraceRecord
from .config import MedCalcContinualConfig
from .pattern_store import MedCalcPatternStore
from .pattern_miner import MedCalcPatternMiner
from .python_layer import PythonCalculatorLayer
from .guided_pipeline import GuidedPipeline

# Import base experiment classes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from benchmark.medcalc.experiments.base import BaseExperiment, ExperimentResult
from benchmark.medcalc.config import ExperimentConfig
from benchmark.medcalc.dataset.loader import MedCalcInstance
from benchmark.medcalc.metrics.cost import calculate_cost
from benchmark.medcalc.metrics.accuracy import calculate_accuracy

# Import L4 pipeline for fallback
from benchmark.medcalc.experiments.l4_pipeline import L4PipelineExperiment


class MedCalcContinualExperiment(BaseExperiment):
    """
    MedCalc Continual Learning Experiment (L5 Continual).

    Pipeline:
    1. Try L2 Python layer first (fast, free)
    2. Back off to L4 Pipeline when needed (rule-based, extraction failures)
    3. Learn from L4 successes to improve future L2 attempts

    Key insight: Python controls workflow, LLMs only handle "understanding"
    tasks when Python can't handle them directly.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        continual_config: Optional[MedCalcContinualConfig] = None,
    ):
        """
        Initialize experiment.

        Args:
            config: Base experiment configuration
            continual_config: MedCalc-specific continual learning config
        """
        super().__init__(config)

        # Continual learning configuration
        self.continual_config = continual_config or MedCalcContinualConfig()

        # Core components
        self.python_layer = PythonCalculatorLayer(self.continual_config)
        self.pattern_store = MedCalcPatternStore(self.continual_config)
        self.pattern_miner = MedCalcPatternMiner(
            self.continual_config,
            self.pattern_store,
        )
        self.guided_pipeline = GuidedPipeline(
            self.continual_config,
            self.pattern_store,
            model=config.model,
        )

        # Create L4 experiment instance for fallback (ensures identical behavior)
        self.l4_experiment = L4PipelineExperiment(config)
        self.l4_experiment.setup()

        # Flag to use L4 directly vs guided pipeline
        self.use_l4_fallback = True

        # Metrics tracking
        self._metrics = {
            "python_success": 0,
            "python_attempts": 0,
            "pipeline_backoff": 0,
            "pipeline_success": 0,
            "pipeline_failure": 0,
            "patterns_learned": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        self._backoff_reasons: Dict[str, int] = defaultdict(int)
        self._accuracy_history: List[bool] = []

        # Run counter for decay
        self._run_count = 0

    def setup(self) -> None:
        """Initialize components."""
        # Nothing special needed - components initialize lazily
        self._setup_complete = True

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run L5 Continual on a single instance.

        Flow:
        1. Try Python layer
        2. If backoff needed, run guided pipeline
        3. If pipeline succeeds and learning enabled, mine patterns
        4. Return result
        """
        start_time = time.time()
        self._run_count += 1
        self._metrics["python_attempts"] += 1

        trace = {
            "method": "l5_continual",
            "stages": [],
        }

        # =====================================================================
        # Stage 1: Try Python Layer (L2)
        # =====================================================================
        python_result = self.python_layer.try_calculate(
            question=instance.question,
            patient_note=instance.patient_note,
        )

        trace["stages"].append({
            "stage": "python_layer",
            "success": python_result.success,
            "calculator": python_result.identified_entity,
            "backoff_reason": python_result.backoff_reason.value if python_result.backoff_reason else None,
        })

        if python_result.success:
            # Python succeeded - no LLM cost!
            self._metrics["python_success"] += 1

            latency_ms = (time.time() - start_time) * 1000

            # Evaluate accuracy
            accuracy = calculate_accuracy(
                predicted=python_result.result,
                ground_truth=instance.ground_truth_answer,
                lower_limit=instance.lower_limit,
                upper_limit=instance.upper_limit,
                output_type=instance.output_type,
                category=instance.category,
            )

            self._accuracy_history.append(accuracy.is_within_tolerance)

            result = ExperimentResult(
                instance_id=instance.row_number,
                calculator_name=instance.calculator_name,
                category=instance.category,
                predicted_answer=python_result.result,
                ground_truth=instance.ground_truth_answer,
                is_correct_exact=accuracy.is_exact_match,
                is_correct_tolerance=accuracy.is_within_tolerance,
                is_within_limits=accuracy.is_within_limits,
                latency_ms=latency_ms,
                input_tokens=0,  # No LLM tokens!
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                num_steps=1,
                trace=trace,
            )

            # Accumulate result for summary
            self.results.append(result)

            return result

        # =====================================================================
        # Stage 2: Back off to L4 Pipeline (same as L4 baseline)
        # =====================================================================
        self._metrics["pipeline_backoff"] += 1
        if python_result.backoff_reason:
            self._backoff_reasons[python_result.backoff_reason.value] += 1

        if self.use_l4_fallback:
            # Use L4 experiment directly for identical behavior
            l4_result = self.l4_experiment.run_instance(instance)

            # Convert L4 result to pipeline_result format
            pipeline_result = LayerResult(
                success=l4_result.is_correct_tolerance or l4_result.predicted_answer is not None,
                result=l4_result.predicted_answer,
                identified_entity=l4_result.calculator_name,
                extracted_values=l4_result.trace.get("stages", [{}])[-1].get("cleaned", {}) if l4_result.trace else {},
                input_tokens=l4_result.input_tokens,
                output_tokens=l4_result.output_tokens,
            )

            input_tokens = l4_result.input_tokens
            output_tokens = l4_result.output_tokens
        else:
            # Use our guided pipeline
            pipeline_result = self.guided_pipeline.run(
                question=instance.question,
                patient_note=instance.patient_note,
                calculator_hint=python_result.identified_entity,
            )
            input_tokens, output_tokens = self.guided_pipeline.get_tokens()

        self._metrics["total_input_tokens"] += input_tokens
        self._metrics["total_output_tokens"] += output_tokens

        trace["stages"].append({
            "stage": "l4_pipeline" if self.use_l4_fallback else "guided_pipeline",
            "success": pipeline_result.success,
            "calculator": pipeline_result.identified_entity,
            "extracted": pipeline_result.extracted_values,
        })

        latency_ms = (time.time() - start_time) * 1000

        # Calculate cost
        cost_metrics = calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
        )

        # Evaluate accuracy
        accuracy = calculate_accuracy(
            predicted=pipeline_result.result,
            ground_truth=instance.ground_truth_answer,
            lower_limit=instance.lower_limit,
            upper_limit=instance.upper_limit,
            output_type=instance.output_type,
            category=instance.category,
        )

        self._accuracy_history.append(accuracy.is_within_tolerance)

        # =====================================================================
        # Stage 3: Learn from Pipeline Success
        # =====================================================================
        if pipeline_result.success and accuracy.is_within_tolerance:
            self._metrics["pipeline_success"] += 1

            # Learn from successful backoff
            if self.continual_config.learning_enabled:
                self._learn_from_success(
                    instance=instance,
                    python_result=python_result,
                    pipeline_result=pipeline_result,
                )
        else:
            self._metrics["pipeline_failure"] += 1

        # =====================================================================
        # Periodic decay
        # =====================================================================
        if (self.continual_config.decay_enabled and
            self._run_count % self.continual_config.decay_interval_runs == 0):
            self._run_decay_cycle()

        result = ExperimentResult(
            instance_id=instance.row_number,
            calculator_name=instance.calculator_name,
            category=instance.category,
            predicted_answer=pipeline_result.result,
            ground_truth=instance.ground_truth_answer,
            is_correct_exact=accuracy.is_exact_match,
            is_correct_tolerance=accuracy.is_within_tolerance,
            is_within_limits=accuracy.is_within_limits,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_metrics.cost_usd,
            num_steps=len(trace["stages"]),
            patterns_used=self._count_patterns_for_calculator(
                pipeline_result.identified_entity
            ),
            patterns_learned=self._metrics["patterns_learned"],
            trace=trace,
        )

        # Accumulate result for summary
        self.results.append(result)

        return result

    def _learn_from_success(
        self,
        instance: MedCalcInstance,
        python_result: LayerResult,
        pipeline_result: LayerResult,
    ) -> None:
        """Record successful backoff for pattern mining."""
        self.pattern_miner.add_successful_backoff(
            question=instance.question,
            patient_note=instance.patient_note,
            calculator=pipeline_result.identified_entity or instance.calculator_name,
            extracted_values=pipeline_result.extracted_values or {},
            ground_truth=instance.ground_truth_answer,
            backoff_reason=python_result.backoff_reason,
        )

        # Check if mining happened
        pending_before = self.pattern_miner.get_pending_count()
        # Mining happens automatically in add_trace when threshold reached
        pending_after = self.pattern_miner.get_pending_count()

        if pending_after < pending_before:
            # Patterns were mined
            self._metrics["patterns_learned"] = self.pattern_store.get_stats()["total_patterns"]

    def _count_patterns_for_calculator(self, calculator: Optional[str]) -> int:
        """Count available patterns for a calculator."""
        if not calculator:
            return 0
        return len(self.pattern_store.get_patterns_for_entity(calculator))

    def _run_decay_cycle(self) -> None:
        """Run pattern decay and pruning."""
        self.pattern_store.apply_decay()
        self.pattern_store.prune_low_confidence(self.continual_config.prune_threshold)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary with continual learning metrics."""
        summary = super().get_summary()

        # Add continual learning specific metrics
        total_attempts = self._metrics["python_attempts"]

        summary.update({
            # Layer usage
            "python_success_count": self._metrics["python_success"],
            "python_success_pct": self._metrics["python_success"] / max(total_attempts, 1) * 100,
            "pipeline_backoff_count": self._metrics["pipeline_backoff"],
            "pipeline_backoff_pct": self._metrics["pipeline_backoff"] / max(total_attempts, 1) * 100,

            # Pipeline results when backed off
            "pipeline_success_count": self._metrics["pipeline_success"],
            "pipeline_success_rate": self._metrics["pipeline_success"] / max(self._metrics["pipeline_backoff"], 1) * 100,

            # Backoff reasons
            "backoff_reasons": dict(self._backoff_reasons),

            # Learning metrics
            "patterns_learned": self.pattern_store.get_stats()["total_patterns"],
            "pending_traces": self.pattern_miner.get_pending_count(),
            "learning_enabled": self.continual_config.learning_enabled,

            # Token usage (only from pipeline)
            "total_input_tokens": self._metrics["total_input_tokens"],
            "total_output_tokens": self._metrics["total_output_tokens"],
        })

        return summary

    def get_learning_curve(self) -> List[float]:
        """Get accuracy over time (rolling window)."""
        if len(self._accuracy_history) < 10:
            return []

        window_size = 10
        curve = []
        for i in range(window_size, len(self._accuracy_history) + 1):
            window = self._accuracy_history[i - window_size:i]
            curve.append(sum(window) / len(window))

        return curve

    def force_mine_patterns(self) -> int:
        """Force pattern mining regardless of threshold."""
        patterns = self.pattern_miner.mine_pending()
        self._metrics["patterns_learned"] = self.pattern_store.get_stats()["total_patterns"]
        return len(patterns)

    def export_patterns(self, path: Optional[str] = None) -> str:
        """Export learned patterns to file."""
        import json

        export_path = path or self.continual_config.export_path
        if not export_path:
            export_path = str(self.continual_config.get_store_path() / "exported_patterns.json")

        stats = self.pattern_store.get_stats()
        patterns = []

        # Collect all patterns from store
        for pattern_id, pattern in self.pattern_store._patterns.items():
            patterns.append(pattern.to_dict())

        export_data = {
            "stats": stats,
            "patterns": patterns,
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return export_path
