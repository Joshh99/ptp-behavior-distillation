"""
Self-Refine: Iterative refinement with self-feedback for single outputs.

Paper: Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (NeurIPS 2023)

Unlike Critic+Repair (which operates on multi-step WorkflowTraces), Self-Refine operates
on the output of a single ptool/LLM call through a Generate -> Feedback -> Refine loop.

Key insight: Python controls the feedback loop — it decides when to stop iterating.
The LLM provides feedback and refined drafts, but Python parses scores and checks
stopping conditions.

Example:
    >>> from ptool_framework.self_refine import SelfRefiner
    >>> refiner = SelfRefiner(max_iterations=3)
    >>> result = refiner.refine("Draft report...", "Write a patient summary")
    >>> print(result.output)  # Refined version
    >>> print(result.iterations)  # How many rounds
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .llm_backend import call_llm, execute_ptool, LLMResponse
from .ptool import PToolSpec
from .traces import WorkflowTrace, TraceStep, StepStatus


# ============================================================================
# Enums
# ============================================================================

class StopReason(Enum):
    """Why the refinement loop stopped."""
    SATISFACTORY = "satisfactory"
    MAX_ITERATIONS = "max_iterations"
    NO_IMPROVEMENT = "no_improvement"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RefinementStep:
    """One iteration of the feedback-refine loop."""
    iteration: int
    draft: str
    feedback: str
    refined: str
    feedback_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "draft": self.draft,
            "feedback": self.feedback,
            "refined": self.refined,
            "feedback_score": self.feedback_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RefinementStep:
        return cls(
            iteration=data["iteration"],
            draft=data["draft"],
            feedback=data["feedback"],
            refined=data["refined"],
            feedback_score=data.get("feedback_score", 0.0),
        )


@dataclass
class RefinementTrace:
    """Complete refinement session."""
    initial_output: str
    final_output: str
    steps: List[RefinementStep] = field(default_factory=list)
    stop_reason: StopReason = StopReason.MAX_ITERATIONS
    task_description: str = ""

    @property
    def iterations(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_output": self.initial_output,
            "final_output": self.final_output,
            "steps": [s.to_dict() for s in self.steps],
            "stop_reason": self.stop_reason.value,
            "task_description": self.task_description,
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RefinementTrace:
        return cls(
            initial_output=data["initial_output"],
            final_output=data["final_output"],
            steps=[RefinementStep.from_dict(s) for s in data.get("steps", [])],
            stop_reason=StopReason(data.get("stop_reason", "max_iterations")),
            task_description=data.get("task_description", ""),
        )

    def to_workflow_trace(self) -> WorkflowTrace:
        """Convert to WorkflowTrace for standard trace storage."""
        trace = WorkflowTrace(
            goal=f"Self-Refine: {self.task_description}" if self.task_description else "Self-Refine"
        )

        # Initial generation step
        gen_step = TraceStep(
            ptool_name="self_refine_generate",
            args={"task": self.task_description},
            goal="Generate initial output",
        )
        gen_step.status = StepStatus.COMPLETED
        gen_step.result = self.initial_output
        trace.steps.append(gen_step)

        # Feedback+refine steps
        for step in self.steps:
            fb_step = TraceStep(
                ptool_name="self_refine_feedback",
                args={"iteration": step.iteration, "draft": step.draft[:100]},
                goal=f"Feedback iteration {step.iteration}",
            )
            fb_step.status = StepStatus.COMPLETED
            fb_step.result = step.feedback
            trace.steps.append(fb_step)

            ref_step = TraceStep(
                ptool_name="self_refine_refine",
                args={"iteration": step.iteration, "feedback": step.feedback[:100]},
                goal=f"Refine iteration {step.iteration}",
            )
            ref_step.status = StepStatus.COMPLETED
            ref_step.result = step.refined
            trace.steps.append(ref_step)

        return trace


@dataclass
class RefinementResult:
    """Final output from Self-Refine."""
    output: str
    trace: RefinementTrace
    iterations: int = 0
    stop_reason: StopReason = StopReason.MAX_ITERATIONS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "trace": self.trace.to_dict(),
            "iterations": self.iterations,
            "stop_reason": self.stop_reason.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RefinementResult:
        return cls(
            output=data["output"],
            trace=RefinementTrace.from_dict(data["trace"]),
            iterations=data.get("iterations", 0),
            stop_reason=StopReason(data.get("stop_reason", "max_iterations")),
        )


# ============================================================================
# Self-Refiner
# ============================================================================

class SelfRefiner:
    """
    Iterative self-refinement: Generate -> Feedback -> Refine loop.

    Args:
        max_iterations: Maximum feedback-refine iterations
        satisfaction_threshold: Score (0-10) at which to stop
        model: LLM model for feedback and refinement
        llm_backend: Custom LLM backend (for testing)
        echo: Whether to print progress
        feedback_fn: Custom feedback function (draft, task) -> (feedback, score)
        stop_fn: Custom stopping function (steps) -> bool
    """

    def __init__(
        self,
        max_iterations: int = 3,
        satisfaction_threshold: float = 8.0,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
        feedback_fn: Optional[Callable[[str, str], tuple]] = None,
        stop_fn: Optional[Callable[[List[RefinementStep]], bool]] = None,
    ):
        self.max_iterations = max_iterations
        self.satisfaction_threshold = satisfaction_threshold
        self.model = model
        self.llm_backend = llm_backend
        self.echo = echo
        self.feedback_fn = feedback_fn
        self.stop_fn = stop_fn

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via backend or default call_llm."""
        if self.llm_backend:
            result = self.llm_backend(prompt, self.model)
            return result.content if isinstance(result, LLMResponse) else result
        return call_llm(prompt, self.model).content

    def refine(self, initial_output: str, task_description: str) -> RefinementResult:
        """
        Iteratively refine an output through feedback.

        Args:
            initial_output: The initial draft to refine
            task_description: Description of what the output should achieve

        Returns:
            RefinementResult with refined output and full trace
        """
        current_draft = initial_output
        steps: List[RefinementStep] = []
        stop_reason = StopReason.MAX_ITERATIONS

        for i in range(self.max_iterations):
            if self.echo:
                print(f"  [Self-Refine] Iteration {i + 1}/{self.max_iterations}...")

            # Get feedback
            if self.feedback_fn:
                feedback, score = self.feedback_fn(current_draft, task_description)
            else:
                feedback, score = self._get_feedback(current_draft, task_description)

            # Check if satisfactory
            if score >= self.satisfaction_threshold:
                steps.append(RefinementStep(
                    iteration=i + 1,
                    draft=current_draft,
                    feedback=feedback,
                    refined=current_draft,
                    feedback_score=score,
                ))
                stop_reason = StopReason.SATISFACTORY
                break

            # Check custom stop condition
            if self.stop_fn and steps and self.stop_fn(steps):
                stop_reason = StopReason.SATISFACTORY
                break

            # Refine based on feedback
            refined = self._refine_draft(current_draft, feedback, task_description)

            # Check for no improvement
            if refined.strip() == current_draft.strip():
                steps.append(RefinementStep(
                    iteration=i + 1,
                    draft=current_draft,
                    feedback=feedback,
                    refined=refined,
                    feedback_score=score,
                ))
                stop_reason = StopReason.NO_IMPROVEMENT
                break

            steps.append(RefinementStep(
                iteration=i + 1,
                draft=current_draft,
                feedback=feedback,
                refined=refined,
                feedback_score=score,
            ))

            current_draft = refined

        final_output = steps[-1].refined if steps else initial_output

        trace = RefinementTrace(
            initial_output=initial_output,
            final_output=final_output,
            steps=steps,
            stop_reason=stop_reason,
            task_description=task_description,
        )

        return RefinementResult(
            output=final_output,
            trace=trace,
            iterations=len(steps),
            stop_reason=stop_reason,
        )

    def refine_ptool(self, spec: PToolSpec, inputs: Dict[str, Any]) -> RefinementResult:
        """
        Generate initial output via ptool, then refine it.

        Args:
            spec: The ptool specification
            inputs: Dictionary of input arguments

        Returns:
            RefinementResult with refined ptool output
        """
        # Generate initial output
        prompt = spec.format_prompt(**inputs)
        initial_output = self._call_llm(prompt)
        task_description = spec.docstring or spec.name
        return self.refine(initial_output.strip(), task_description)

    def _get_feedback(self, draft: str, task_description: str) -> tuple:
        """
        Get feedback on a draft from the LLM.

        Returns:
            (feedback_text, score) where score is 0-10.
        """
        prompt = (
            f"Task: {task_description}\n\n"
            f"Current output:\n{draft}\n\n"
            f"Provide feedback to improve this output. Rate it on a scale of 0-10.\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"feedback": "<specific improvements needed>", "score": <0-10>}}'
        )
        response = self._call_llm(prompt)

        # Parse feedback
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                feedback = data.get("feedback", response)
                score = float(data.get("score", 5))
            else:
                feedback = response.strip()
                score_match = re.search(r'(\d+(?:\.\d+)?)\s*/?\s*10', response)
                score = float(score_match.group(1)) if score_match else 5.0
        except (json.JSONDecodeError, ValueError):
            feedback = response.strip()
            score = 5.0

        return feedback, score

    def _refine_draft(self, draft: str, feedback: str, task_description: str) -> str:
        """Refine a draft given feedback."""
        prompt = (
            f"Task: {task_description}\n\n"
            f"Current draft:\n{draft}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Please provide an improved version that addresses the feedback.\n"
            f"Output ONLY the improved text, no explanations."
        )
        return self._call_llm(prompt).strip()


# ============================================================================
# Convenience Functions
# ============================================================================

def refine(
    output: str,
    task_description: str,
    max_iterations: int = 3,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
    echo: bool = False,
) -> RefinementResult:
    """
    Quick self-refinement of an output.

    Args:
        output: The initial output to refine
        task_description: What the output should achieve
        max_iterations: Maximum feedback-refine iterations
        model: LLM model
        llm_backend: Custom backend (for testing)
        echo: Print progress

    Returns:
        RefinementResult with refined output
    """
    refiner = SelfRefiner(
        max_iterations=max_iterations,
        model=model,
        llm_backend=llm_backend,
        echo=echo,
    )
    return refiner.refine(output, task_description)
