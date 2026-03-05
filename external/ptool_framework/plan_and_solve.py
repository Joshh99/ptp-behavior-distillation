"""
Plan-and-Solve: Two-phase agent with explicit planning and deterministic execution.

Paper: Wang et al., "Plan-and-Solve Prompting" (ACL 2023)

Phase 1 (Plan): LLM generates a complete plan as a list of subtasks.
Phase 2 (Solve): Python executes each subtask sequentially, with optional replanning
on failure.

Fills the gap between:
- ReAct (dynamic, one step at a time)
- WorkflowTrace (static, pre-defined)

Key insight: The plan is inspectable and modifiable before execution. Python validates
ptool names, checks for cycles, and handles replanning — LLM only generates plans
and executes subtasks.

Example:
    >>> from ptool_framework.plan_and_solve import PlanAndSolve
    >>> agent = PlanAndSolve(available_ptools=[...])
    >>> result = agent.run("Calculate BMI for 70kg, 175cm patient")
    >>> print(result.plan.subtasks)
    >>> print(result.final_result)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .llm_backend import call_llm, execute_ptool, LLMResponse
from .ptool import PToolSpec, get_registry
from .traces import WorkflowTrace, TraceStep, StepStatus


# ============================================================================
# Data Structures
# ============================================================================

class SubtaskStatus(Enum):
    """Status of a subtask in the plan."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A single step in a plan."""
    description: str
    ptool_name: Optional[str] = None  # None = pure LLM step
    args_template: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)  # indices of prerequisite subtasks
    result: Optional[Any] = None
    status: SubtaskStatus = SubtaskStatus.PENDING
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "ptool_name": self.ptool_name,
            "args_template": self.args_template,
            "depends_on": self.depends_on,
            "result": self.result,
            "status": self.status.value,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Subtask:
        return cls(
            description=data["description"],
            ptool_name=data.get("ptool_name"),
            args_template=data.get("args_template", {}),
            depends_on=data.get("depends_on", []),
            result=data.get("result"),
            status=SubtaskStatus(data.get("status", "pending")),
            error=data.get("error"),
        )


@dataclass
class Plan:
    """A complete plan with subtasks."""
    goal: str
    subtasks: List[Subtask] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(s.status == SubtaskStatus.COMPLETED for s in self.subtasks)

    @property
    def has_failures(self) -> bool:
        return any(s.status == SubtaskStatus.FAILED for s in self.subtasks)

    def completed_results(self) -> Dict[int, Any]:
        """Map of subtask index -> result for completed subtasks."""
        return {
            i: s.result for i, s in enumerate(self.subtasks)
            if s.status == SubtaskStatus.COMPLETED
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "subtasks": [s.to_dict() for s in self.subtasks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Plan:
        return cls(
            goal=data["goal"],
            subtasks=[Subtask.from_dict(s) for s in data.get("subtasks", [])],
        )

    def to_workflow_trace(self) -> WorkflowTrace:
        """Convert to WorkflowTrace for standard trace storage/distillation."""
        trace = WorkflowTrace(goal=self.goal)
        for i, subtask in enumerate(self.subtasks):
            step = TraceStep(
                ptool_name=subtask.ptool_name or "llm_step",
                args=subtask.args_template,
                goal=subtask.description,
            )
            if subtask.status == SubtaskStatus.COMPLETED:
                step.status = StepStatus.COMPLETED
                step.result = subtask.result
            elif subtask.status == SubtaskStatus.FAILED:
                step.status = StepStatus.FAILED
                step.error = subtask.error
            trace.steps.append(step)
        return trace


@dataclass
class PlanExecutionResult:
    """Result of plan execution."""
    plan: Plan
    success: bool
    final_result: Optional[Any] = None
    replanned: bool = False
    replan_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "success": self.success,
            "final_result": self.final_result,
            "replanned": self.replanned,
            "replan_count": self.replan_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanExecutionResult:
        return cls(
            plan=Plan.from_dict(data["plan"]),
            success=data["success"],
            final_result=data.get("final_result"),
            replanned=data.get("replanned", False),
            replan_count=data.get("replan_count", 0),
        )


# ============================================================================
# Plan-and-Solve Agent
# ============================================================================

class PlanAndSolve:
    """
    Two-phase agent: plan first, then execute.

    Args:
        available_ptools: List of PToolSpec instances the agent can use
        model: LLM model for planning and execution
        llm_backend: Custom LLM backend (for testing)
        echo: Whether to print progress
        allow_replan: Whether to replan on subtask failure
        max_replan_attempts: Maximum number of replanning attempts
    """

    def __init__(
        self,
        available_ptools: Optional[List[PToolSpec]] = None,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
        allow_replan: bool = True,
        max_replan_attempts: int = 2,
    ):
        self.available_ptools = available_ptools or []
        self.model = model
        self.llm_backend = llm_backend
        self.echo = echo
        self.allow_replan = allow_replan
        self.max_replan_attempts = max_replan_attempts

        # Build ptool lookup
        self._ptool_map: Dict[str, PToolSpec] = {
            spec.name: spec for spec in self.available_ptools
        }

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via backend or default call_llm."""
        if self.llm_backend:
            result = self.llm_backend(prompt, self.model)
            return result.content if isinstance(result, LLMResponse) else result
        return call_llm(prompt, self.model).content

    def plan(self, goal: str) -> Plan:
        """
        Generate a plan for achieving a goal.

        Args:
            goal: What to accomplish

        Returns:
            Plan with list of subtasks
        """
        # Build ptool descriptions for the planner
        ptool_descriptions = ""
        if self.available_ptools:
            ptool_list = []
            for spec in self.available_ptools:
                params = ", ".join(f"{k}: {v.__name__}" for k, v in spec.parameters.items())
                ptool_list.append(f"  - {spec.name}({params}): {spec.docstring}")
            ptool_descriptions = f"\nAvailable tools:\n" + "\n".join(ptool_list) + "\n"

        prompt = (
            f"Create a step-by-step plan to achieve the following goal.\n\n"
            f"Goal: {goal}\n"
            f"{ptool_descriptions}\n"
            f"Respond with ONLY a JSON array of subtasks:\n"
            f'[{{"description": "...", "ptool_name": "tool_name_or_null", '
            f'"args_template": {{}}, "depends_on": []}}]\n\n'
            f"Each subtask should have:\n"
            f"- description: what this step does\n"
            f"- ptool_name: name of the tool to use (null for pure LLM reasoning)\n"
            f"- args_template: arguments for the tool\n"
            f"- depends_on: list of step indices (0-based) this step depends on"
        )

        if self.echo:
            print(f"  [Plan-and-Solve] Generating plan for: {goal}")

        response = self._call_llm(prompt)

        # Parse plan
        plan = self._parse_plan(response, goal)

        if self.echo:
            print(f"  [Plan-and-Solve] Plan has {len(plan.subtasks)} subtasks")
            for i, st in enumerate(plan.subtasks):
                print(f"    {i + 1}. {st.description} (tool: {st.ptool_name or 'LLM'})")

        return plan

    def _parse_plan(self, response: str, goal: str) -> Plan:
        """Parse LLM response into a Plan."""
        # Extract JSON array
        try:
            # Try to find JSON array in response
            array_match = re.search(r'\[[\s\S]*\]', response)
            if array_match:
                subtask_data = json.loads(array_match.group())
            else:
                subtask_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: single subtask with the entire response
            return Plan(
                goal=goal,
                subtasks=[Subtask(description=response.strip())],
            )

        subtasks = []
        for item in subtask_data:
            if isinstance(item, dict):
                subtasks.append(Subtask(
                    description=item.get("description", ""),
                    ptool_name=item.get("ptool_name"),
                    args_template=item.get("args_template", {}),
                    depends_on=item.get("depends_on", []),
                ))

        return Plan(goal=goal, subtasks=subtasks)

    def validate_plan(self, plan: Plan) -> List[str]:
        """
        Validate a plan before execution.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for i, subtask in enumerate(plan.subtasks):
            # Check ptool exists
            if subtask.ptool_name and subtask.ptool_name not in self._ptool_map:
                errors.append(
                    f"Subtask {i}: unknown ptool '{subtask.ptool_name}'. "
                    f"Available: {list(self._ptool_map.keys())}"
                )

            # Check dependencies are valid indices
            for dep in subtask.depends_on:
                if dep < 0 or dep >= len(plan.subtasks):
                    errors.append(f"Subtask {i}: invalid dependency index {dep}")
                if dep >= i:
                    errors.append(f"Subtask {i}: dependency {dep} is not a predecessor")

        return errors

    def execute_plan(self, plan: Plan) -> PlanExecutionResult:
        """
        Execute a plan, running each subtask in order.

        Args:
            plan: The plan to execute

        Returns:
            PlanExecutionResult with outcomes
        """
        replan_count = 0

        for i, subtask in enumerate(plan.subtasks):
            # Check dependencies are met
            for dep in subtask.depends_on:
                if plan.subtasks[dep].status != SubtaskStatus.COMPLETED:
                    subtask.status = SubtaskStatus.SKIPPED
                    subtask.error = f"Dependency {dep} not completed"
                    continue

            subtask.status = SubtaskStatus.RUNNING
            if self.echo:
                print(f"  [Plan-and-Solve] Executing step {i + 1}: {subtask.description}")

            try:
                result = self._execute_subtask(subtask, plan)
                subtask.result = result
                subtask.status = SubtaskStatus.COMPLETED
            except Exception as e:
                subtask.status = SubtaskStatus.FAILED
                subtask.error = str(e)

                if self.echo:
                    print(f"  [Plan-and-Solve] Step {i + 1} failed: {e}")

                # Attempt replan
                if self.allow_replan and replan_count < self.max_replan_attempts:
                    if self.echo:
                        print(f"  [Plan-and-Solve] Replanning (attempt {replan_count + 1})...")
                    new_plan = self._replan(plan, i)
                    replan_count += 1

                    # Replace remaining subtasks
                    plan.subtasks = plan.subtasks[:i + 1] + new_plan.subtasks
                    # Continue execution with next subtask
                    continue

        # Determine final result from last completed subtask
        final_result = None
        for subtask in reversed(plan.subtasks):
            if subtask.status == SubtaskStatus.COMPLETED:
                final_result = subtask.result
                break

        return PlanExecutionResult(
            plan=plan,
            success=plan.is_complete,
            final_result=final_result,
            replanned=replan_count > 0,
            replan_count=replan_count,
        )

    def _execute_subtask(self, subtask: Subtask, plan: Plan) -> Any:
        """Execute a single subtask."""
        # Resolve argument templates (replace $step_N references)
        resolved_args = self._resolve_args(subtask.args_template, plan)

        if subtask.ptool_name and subtask.ptool_name in self._ptool_map:
            # Execute via ptool
            spec = self._ptool_map[subtask.ptool_name]
            return execute_ptool(
                spec, resolved_args,
                custom_backend=self.llm_backend,
                collect_traces=False,
            )
        else:
            # Pure LLM step
            context_parts = [f"Task: {subtask.description}"]
            if resolved_args:
                context_parts.append(f"Context: {json.dumps(resolved_args)}")
            prompt = "\n".join(context_parts)
            return self._call_llm(prompt)

    def _resolve_args(self, args_template: Dict[str, Any], plan: Plan) -> Dict[str, Any]:
        """Resolve $step_N references in argument templates."""
        resolved = {}
        completed = plan.completed_results()

        for key, value in args_template.items():
            if isinstance(value, str) and value.startswith("$step_"):
                # Extract step index
                try:
                    idx = int(value.replace("$step_", ""))
                    resolved[key] = completed.get(idx, value)
                except ValueError:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def _replan(self, plan: Plan, failed_step: int) -> Plan:
        """Generate a new plan from partial progress after a failure."""
        completed = []
        for i, s in enumerate(plan.subtasks[:failed_step]):
            if s.status == SubtaskStatus.COMPLETED:
                completed.append(f"Step {i + 1} ({s.description}): {s.result}")

        failed = plan.subtasks[failed_step]
        prompt = (
            f"The following plan partially failed. Create a revised plan to complete the goal.\n\n"
            f"Goal: {plan.goal}\n\n"
            f"Completed steps:\n" + "\n".join(completed) + "\n\n"
            f"Failed step: {failed.description} (error: {failed.error})\n\n"
            f"Create new subtasks to complete the goal from this point.\n"
            f"Respond with ONLY a JSON array of subtasks:\n"
            f'[{{"description": "...", "ptool_name": null, "args_template": {{}}, "depends_on": []}}]'
        )

        response = self._call_llm(prompt)
        return self._parse_plan(response, plan.goal)

    def run(self, goal: str) -> PlanExecutionResult:
        """
        Full plan-and-solve: generate plan, validate, execute.

        Args:
            goal: What to accomplish

        Returns:
            PlanExecutionResult with final outcome
        """
        plan = self.plan(goal)
        errors = self.validate_plan(plan)
        if errors and self.echo:
            print(f"  [Plan-and-Solve] Validation warnings: {errors}")
        return self.execute_plan(plan)


# ============================================================================
# Convenience Functions
# ============================================================================

def plan_and_solve(
    goal: str,
    available_ptools: Optional[List[PToolSpec]] = None,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
    echo: bool = False,
) -> PlanExecutionResult:
    """
    Quick plan-and-solve execution.

    Args:
        goal: What to accomplish
        available_ptools: Available tools
        model: LLM model
        llm_backend: Custom backend (for testing)
        echo: Print progress

    Returns:
        PlanExecutionResult with outcome
    """
    agent = PlanAndSolve(
        available_ptools=available_ptools,
        model=model,
        llm_backend=llm_backend,
        echo=echo,
    )
    return agent.run(goal)
