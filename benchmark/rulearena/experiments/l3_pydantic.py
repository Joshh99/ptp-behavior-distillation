"""
L3-PydanticAI Experiment: PydanticAI native tool-calling agent for RuleArena.

Robustness check at the same spectrum position as l3_react.py (L3).
Replaces the custom ReActAgent text-parsing loop in ptool_framework with
PydanticAI's native OpenAI function-calling loop over the same underlying tools.

Design decisions vs l3_react.py
--------------------------------
- Same tools, same goal builders, same calculator wrappers — all imported
  directly from l3_react.py so there is no behavioural drift.
- PydanticAI manages the think/act/observe loop internally via OpenAI tool
  calls; no <thought>/<action>/<answer> text parsing required.
- One domain-specific Agent is created per domain at module load time.
  Tools are registered with @agent.tool_plain (no RunContext dependency).

Output schema
-------------
Identical to run_l3_instance() in l3_react.py:
  answer, success, trajectory, termination_reason, total_llm_calls, total_time_ms

total_llm_calls counting method
---------------------------------
PydanticAI does not expose a single "total LLM calls" counter.  We derive it
as follows and document both components in the trajectory metadata:

  pydantic_llm_calls = number of ModelResponse objects in result.all_messages()
                       (each ModelResponse = one LLM API round-trip managed by
                        PydanticAI's orchestration loop)

  extraction_llm_calls = number of tool calls whose name starts with
                         "extract_" (each such call invokes the ptool
                          PToolWrapper.__call__ → execute_ptool → one LLM call)

  total_llm_calls = pydantic_llm_calls + extraction_llm_calls

Token counting
--------------
result.usage() covers PydanticAI's orchestration calls only.
The ptool token accumulator (get_token_accumulator) captures the extraction
tool LLM calls.  Both deltas are summed for the ExperimentResult token fields.
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Windows UTF-8 fix
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    for _s in ("stdout", "stderr"):
        _stream = getattr(sys, _s, None)
        if _stream and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup — must happen before framework imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent.parent
_EXTERNAL = str(REPO_ROOT / "external")
if _EXTERNAL not in sys.path:
    sys.path.insert(0, _EXTERNAL)

from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelResponse, ToolCallPart

from ptool_framework.llm_backend import get_token_accumulator           # noqa: E402

from benchmark.rulearena.config import MODEL_CONFIG, calculate_cost
from benchmark.rulearena.dataset.loader import RuleArenaInstance
from benchmark.rulearena.experiments.base import BaseExperiment, ExperimentResult

# ---------------------------------------------------------------------------
# Import all tools from l3_react.py — no reimplementation
# ---------------------------------------------------------------------------
from benchmark.rulearena.experiments.l3_react import (
    # LLM extraction ptools (PToolWrapper objects; __call__ invokes LLM)
    extract_airline_params,
    extract_tax_params,
    extract_nba_params,
    # Python calculator wrappers
    compute_airline_fee_ptool,
    compute_tax_fee_ptool,
    # Goal string builders (same prompt engineering as l3_react)
    _build_airline_goal,
    _build_tax_goal,
    _build_nba_goal,
)

# ---------------------------------------------------------------------------
# PydanticAI model setup
# ---------------------------------------------------------------------------
_client = AsyncOpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)
_pydantic_model = OpenAIModel(
    MODEL_CONFIG["model_id"],
    provider=OpenAIProvider(openai_client=_client),
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a rule compliance and fee calculation agent. "
    "Use the available tools step by step following the instructions in the query. "
    "After all tool calls are complete, respond with ONLY the final answer:\n"
    "  - Fee/tax calculations: a plain integer or decimal number (no $ symbol)\n"
    "  - NBA compliance checks: exactly True or False\n"
    "No explanation — just the bare answer value."
)

# ---------------------------------------------------------------------------
# Numpy-type sanitiser
# The airline calculator returns numpy.int64; PydanticAI serialises tool
# results to JSON so all return values must be Python-native.
# ---------------------------------------------------------------------------

def _to_python(obj: Any) -> Any:
    """Recursively coerce numpy scalars and similar to Python native types."""
    if hasattr(obj, "item"):          # numpy scalar (int64, float64, …)
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Domain-specific agents
# Each agent has exactly the tools relevant to its domain.
# Agents are created once at module load time and reused across instances.
# ---------------------------------------------------------------------------

# --- Airline agent ---
_airline_agent: Agent[None, str] = Agent(
    model=_pydantic_model,
    output_type=str,
    system_prompt=_SYSTEM_PROMPT,
)


@_airline_agent.tool_plain
def extract_airline_params_fn(query: str) -> dict:  # type: ignore[return]
    """
    Extract structured baggage parameters from an airline query.

    Returns a dict with: base_price, customer_class, routine, direction, bag_list.
    Pass the COMPLETE returned dict directly to compute_airline_fee_fn — do not
    reconstruct or omit any fields.
    """
    return _to_python(extract_airline_params(query=query))  # type: ignore[call-arg]


@_airline_agent.tool_plain
def compute_airline_fee_fn(params: dict) -> dict:  # type: ignore[return]
    """
    Compute total airline cost (ticket price + baggage fees).

    Pass the FULL dict returned by extract_airline_params_fn as params.
    Returns {"result": <total cost as int>, "calculator": "airline_baggage_fee"}.
    """
    return _to_python(compute_airline_fee_ptool(params))


# --- Tax agent ---
_tax_agent: Agent[None, str] = Agent(
    model=_pydantic_model,
    output_type=str,
    system_prompt=_SYSTEM_PROMPT,
)


@_tax_agent.tool_plain
def extract_tax_params_fn(query: str) -> dict:  # type: ignore[return]
    """
    Extract TaxPayer fields from filled IRS forms text.

    Returns a large dict covering Form 1040, Schedule 1/2/3, and optionally
    Schedule A, Schedule C, and Form 8863.
    Pass the COMPLETE returned dict directly to compute_tax_fee_fn — do not
    reconstruct, omit, or modify any fields.
    """
    return _to_python(extract_tax_params(query=query))  # type: ignore[call-arg]


@_tax_agent.tool_plain
def compute_tax_fee_fn(params: dict) -> dict:  # type: ignore[return]
    """
    Compute federal tax owed (or refund) from extracted TaxPayer parameters.

    Pass the FULL dict returned by extract_tax_params_fn as params.
    Returns {"result": <amount as float>, "calculator": "tax_fee"}.
    Positive = owed. Negative = refund.
    """
    return _to_python(compute_tax_fee_ptool(params))


# --- NBA agent ---
_nba_agent: Agent[None, str] = Agent(
    model=_pydantic_model,
    output_type=str,
    system_prompt=_SYSTEM_PROMPT,
)


@_nba_agent.tool_plain
def extract_nba_params_fn(query: str) -> dict:  # type: ignore[return]
    """
    Determine NBA CBA salary-cap rule compliance for the described operations.

    Returns {"verdict": <bool>, "illegal_operation": <str>, "problematic_team": <str>,
             "reasoning": <str>}.
    The verdict field is the final answer. Output "True" if any operation violates
    the rules, "False" if all are compliant.
    """
    return _to_python(extract_nba_params(query=query))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Domain → agent mapping  &  goal builders
# ---------------------------------------------------------------------------
_DOMAIN_AGENTS: Dict[str, Agent] = {
    "airline": _airline_agent,
    "tax": _tax_agent,
    "nba": _nba_agent,
}

_GOAL_BUILDERS = {
    "airline": _build_airline_goal,
    "tax": _build_tax_goal,
    "nba": _build_nba_goal,
}

# Names of extraction tool functions — each invocation = 1 additional LLM call
_EXTRACTION_TOOL_NAMES = {
    "extract_airline_params_fn",
    "extract_tax_params_fn",
    "extract_nba_params_fn",
}


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------

def _build_trajectory(
    goal: str,
    messages: list,
    answer: Optional[str],
    success: bool,
    termination_reason: str,
    total_llm_calls: int,
    pydantic_llm_calls: int,
    extraction_llm_calls: int,
    total_time_ms: float,
) -> Dict[str, Any]:
    """
    Convert PydanticAI message history into a trajectory dict.

    The structure mirrors ReActTrajectory.to_dict() where possible so that
    downstream analysis code can treat both L3 variants the same way.
    Fields that have no PydanticAI equivalent are set to None with an
    explanatory comment in the metadata.
    """
    steps = []
    for msg in messages:
        step: Dict[str, Any] = {
            "type": type(msg).__name__,
            "parts": [],
        }
        if hasattr(msg, "parts"):
            for part in msg.parts:
                p: Dict[str, Any] = {"type": type(part).__name__}
                # Text content (model reasoning / final answer)
                if hasattr(part, "content") and part.content:
                    p["content"] = str(part.content)[:600]
                # Tool call
                if hasattr(part, "tool_name"):
                    p["tool_name"] = part.tool_name
                if hasattr(part, "args"):
                    p["args"] = str(part.args)[:600]
                # Tool return value
                if hasattr(part, "content") and hasattr(part, "tool_name"):
                    # ToolReturnPart has both tool_name and content
                    pass  # already captured above
                step["parts"].append(p)
        steps.append(step)

    return {
        "trajectory_id": str(uuid.uuid4())[:8],
        "goal": goal[:300],
        "steps": steps,
        "final_answer": answer,
        "success": success,
        "termination_reason": termination_reason,
        # No separate generated_trace — PydanticAI uses native function calling,
        # not ptool_framework WorkflowTrace.
        "generated_trace": None,
        "model_used": f"{MODEL_CONFIG['model_id']} (PydanticAI)",
        "total_llm_calls": total_llm_calls,
        "total_llm_calls_breakdown": {
            "pydantic_orchestration": pydantic_llm_calls,
            "extraction_tool_calls": extraction_llm_calls,
            "note": (
                "pydantic_orchestration = ModelResponse count in all_messages(); "
                "extraction_tool_calls = ToolCallPart count with extract_* names "
                "(each invokes one ptool LLM call)"
            ),
        },
        "total_time_ms": total_time_ms,
        "created_at": datetime.now().isoformat(),
        # ptp_trace: not applicable — PydanticAI uses native JSON tool calls,
        # not the text-based <thought>/<action>/<answer> format used by ReActAgent.
        "ptp_trace": None,
    }


# ---------------------------------------------------------------------------
# run_l3_pydantic_instance
# ---------------------------------------------------------------------------

def run_l3_pydantic_instance(
    problem: RuleArenaInstance,
    domain: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the PydanticAI agent on a single RuleArena instance.

    Returns the same keys as run_l3_instance() in l3_react.py:
        answer, success, trajectory, termination_reason,
        total_llm_calls, total_time_ms
    """
    if domain not in _DOMAIN_AGENTS:
        raise ValueError(f"Unknown domain: {domain!r}. Expected airline, tax, or nba.")

    agent = _DOMAIN_AGENTS[domain]
    goal = _GOAL_BUILDERS[domain](problem)

    start_time = time.time()
    termination_reason = "answer_found"
    answer: Optional[str] = None
    success = False

    try:
        result = agent.run_sync(goal)

        raw_output: str = result.output or ""

        # Best-effort answer extraction:
        # 1. Check for <answer> tags in case the model added them despite instructions
        tag_match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL)
        if tag_match:
            answer = tag_match.group(1).strip()
        else:
            answer = raw_output.strip()

        success = bool(answer)
        if not success:
            termination_reason = "error"

        messages = result.all_messages()

        # Count LLM calls
        pydantic_llm_calls = sum(1 for m in messages if isinstance(m, ModelResponse))
        extraction_llm_calls = sum(
            1
            for m in messages
            if isinstance(m, ModelResponse)
            for p in m.parts
            if isinstance(p, ToolCallPart) and p.tool_name in _EXTRACTION_TOOL_NAMES
        )
        total_llm_calls = pydantic_llm_calls + extraction_llm_calls

        total_time_ms = (time.time() - start_time) * 1000

        if debug:
            print(f"\n[PydanticAI] raw_output: {raw_output!r}")
            print(f"[PydanticAI] answer: {answer!r}")
            print(f"[PydanticAI] llm_calls: {total_llm_calls} "
                  f"(pydantic={pydantic_llm_calls}, extraction={extraction_llm_calls})")

        trajectory = _build_trajectory(
            goal=goal,
            messages=messages,
            answer=answer,
            success=success,
            termination_reason=termination_reason,
            total_llm_calls=total_llm_calls,
            pydantic_llm_calls=pydantic_llm_calls,
            extraction_llm_calls=extraction_llm_calls,
            total_time_ms=total_time_ms,
        )

        return {
            "answer": answer,
            "success": success,
            "trajectory": trajectory,
            "termination_reason": termination_reason,
            "total_llm_calls": total_llm_calls,
            "total_time_ms": total_time_ms,
            # Also expose usage for cost accounting in run_instance
            "_usage": result.usage(),
        }

    except Exception as exc:
        total_time_ms = (time.time() - start_time) * 1000
        return {
            "answer": None,
            "success": False,
            "trajectory": {
                "trajectory_id": str(uuid.uuid4())[:8],
                "goal": goal[:300],
                "steps": [],
                "final_answer": None,
                "success": False,
                "termination_reason": "error",
                "generated_trace": None,
                "model_used": f"{MODEL_CONFIG['model_id']} (PydanticAI)",
                "total_llm_calls": 0,
                "total_time_ms": total_time_ms,
                "created_at": datetime.now().isoformat(),
                "ptp_trace": None,
            },
            "termination_reason": "error",
            "total_llm_calls": 0,
            "total_time_ms": total_time_ms,
            "_usage": None,
            "_error": str(exc),
        }


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class L3_Pydantic_Experiment(BaseExperiment):
    """
    L3-PydanticAI: PydanticAI native tool-calling agent.

    Robustness check at the L3 position.  Uses the same tools and goal
    builders as L3_ReAct_Experiment but replaces the custom ReActAgent
    text-parsing loop with PydanticAI's OpenAI function-calling loop.
    """

    def __init__(self):
        super().__init__(
            experiment_name="l3_pydantic",
            model_id=MODEL_CONFIG["model_id"],
        )
        self.debug = MODEL_CONFIG.get("debug", False)

    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        start_time = time.time()
        acc = get_token_accumulator()
        input_before = acc.total_prompt_tokens
        output_before = acc.total_completion_tokens

        try:
            r = run_l3_pydantic_instance(
                problem=instance,
                domain=instance.domain,
                debug=self.debug,
            )

            # Surface inner exception if the helper caught one
            if "_error" in r:
                raise RuntimeError(r["_error"])

            predicted = r["answer"]

            # Coerce NBA boolean strings (same logic as L3_ReAct_Experiment)
            if instance.domain == "nba" and isinstance(predicted, str):
                if predicted.strip().lower() == "true":
                    predicted = True
                elif predicted.strip().lower() == "false":
                    predicted = False

            exact, tolerance = self.compare_answers(predicted, instance.ground_truth_answer)

            # Token accounting: ptool accumulator delta = extraction tool tokens
            ptool_input = acc.total_prompt_tokens - input_before
            ptool_output = acc.total_completion_tokens - output_before

            # PydanticAI orchestration tokens from result.usage()
            usage = r.get("_usage")
            if usage is not None:
                pydantic_input = getattr(usage, "request_tokens", 0) or 0
                pydantic_output = getattr(usage, "response_tokens", 0) or 0
            else:
                pydantic_input = pydantic_output = 0

            total_input = ptool_input + pydantic_input
            total_output = ptool_output + pydantic_output

            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=predicted,
                expected=instance.ground_truth_answer,
                is_correct_exact=exact,
                is_correct_tolerance=tolerance,
                latency_ms=r["total_time_ms"],
                cost_usd=calculate_cost(total_input, total_output),
                input_tokens=total_input,
                output_tokens=total_output,
                calculator_name=instance.domain,
                failure_mode="none" if exact else "reasoning_error",
                metadata={
                    "domain": instance.domain,
                    "total_llm_calls": r["total_llm_calls"],
                    "termination_reason": r["termination_reason"],
                    "trajectory": r["trajectory"],
                    "token_breakdown": {
                        "ptool_input": ptool_input,
                        "ptool_output": ptool_output,
                        "pydantic_input": pydantic_input,
                        "pydantic_output": pydantic_output,
                    },
                },
            )

        except Exception as e:
            ptool_input = acc.total_prompt_tokens - input_before
            ptool_output = acc.total_completion_tokens - output_before
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=calculate_cost(ptool_input, ptool_output),
                input_tokens=ptool_input,
                output_tokens=ptool_output,
                calculator_name=instance.domain,
                error=str(e),
                failure_mode="agent_error",
                metadata={"domain": instance.domain},
            )
