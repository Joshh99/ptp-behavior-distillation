"""
L3 Experiment: ReActAgent with multi-step tool use for RuleArena.

Exports:
  - AIRLINE_CALC_SPEC  : PToolSpec wrapping compute_airline_fee (model="python")
  - TAX_CALC_SPEC      : PToolSpec wrapping compute_tax_fee    (model="python")
  - get_rulearena_ptools(domain) -> List[PToolSpec]
  - run_l3_instance(problem, domain, model, max_steps) -> Dict

The extraction ptools (extract_airline_params, extract_tax_params,
extract_nba_params) are imported from l1_ptool -- not redefined here.
Those wrappers carry a .spec attribute (PToolWrapper.spec) that we pass
directly to ReActAgent.

compute_tax_fee_ptool performs structural adaptation (flat dict ->
{"pydantic": wrapped dict} + default injection) so the agent sees a clean
single-dict interface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import time
from typing import Any, Optional, Dict

# ---------------------------------------------------------------------------
# Fix Windows charmap crash: loguru/print writing Unicode to cp1252 stderr/stdout
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        if _stream and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup — must happen before framework imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent.parent
_EXTERNAL = str(REPO_ROOT / "external")
if _EXTERNAL not in sys.path:
    sys.path.insert(0, _EXTERNAL)

from ptool_framework.ptool import PToolSpec, get_registry          # noqa: E402
from ptool_framework.react import ReActAgent                       # noqa: E402
from ptool_framework.llm_backend import get_token_accumulator      # noqa: E402

from benchmark.rulearena.experiments.base import BaseExperiment, ExperimentResult
from benchmark.rulearena.config import calculate_cost
from benchmark.rulearena.calculators.airline import compute_airline_fee as _airline_calc_raw
from benchmark.rulearena.calculators.tax import compute_tax_fee as _tax_calc_raw
from benchmark.rulearena.config import MODEL_CONFIG
from benchmark.rulearena.dataset.loader import RuleArenaInstance
from benchmark.rulearena.experiments.l1_ptool import (
    extract_airline_params,
    extract_nba_params,
    extract_tax_params,
    _build_nba_query,
)
from benchmark.rulearena.experiments.l0f_cot import build_tax_query

_MODEL = MODEL_CONFIG["model_id"]  # "deepseek-ai/DeepSeek-V3"


# ============================================================================
# Schedule defaults for the tax wrapper
# Mirrors L1_PTool_Experiment._SCHED_* exactly — supply zeros for optional
# schedules so compute_answer() does not crash on attribute lookups.
# ============================================================================

# Required TaxPayer fields that the agent's extraction may omit.
# Applied before schedule-specific defaults so they cannot override a value
# the agent did supply.
_TAXPAYER_REQUIRED_DEFAULTS: Dict[str, Any] = {
    # identity / metadata
    "name": "",
    "filing_status": "single",
    # int
    "age": 0,
    "spouse_age": 0,
    "num_qualifying_children": 0,
    "num_other_dependents": 0,
    # bool
    "blind": False,
    "spouse_blind": False,
    "itemized": False,
    "self_employed": False,
    "has_student_loans_or_education_expenses": False,
    # float — Form 1040 income lines
    "wage_tip_compensation": 0.0,
    "household_employee_wage": 0.0,
    "unreported_tip": 0.0,
    "nontaxable_combat_pay": 0.0,
    "tax_exempt_interest": 0.0,
    "taxable_interest": 0.0,
    "qualified_dividends": 0.0,
    "ordinary_dividends": 0.0,
    "ira_distributions": 0.0,
    "taxable_ira_distributions": 0.0,
    "all_pensions": 0.0,
    "taxable_pensions": 0.0,
    "social_security_benefits": 0.0,
    "taxable_social_security_benefits": 0.0,
    "qualified_business_income": 0.0,
    "federal_income_tax_withheld": 0.0,
    "earned_income_credit": 0.0,
    # float — Schedule 1
    "taxable_state_refunds": 0.0,
    "alimony_income": 0.0,
    "sale_of_business": 0.0,
    "rental_real_estate_sch1": 0.0,
    "farm_income": 0.0,
    "unemployment_compensation": 0.0,
    "other_income": 0.0,
    "educator_expenses": 0.0,
    "hsa_deduction": 0.0,
    "ira_deduction": 0.0,
    "student_loan_interest_deduction": 0.0,
    "other_adjustments": 0.0,
    # float — Schedule 2
    "amt_f6251": 0.0,
    "credit_repayment": 0.0,
    "other_additional_taxes": 0.0,
    # float — Schedule 3
    "foreign_tax_credit": 0.0,
    "dependent_care": 0.0,
    "retirement_savings": 0.0,
    "elderly_disabled_credits": 0.0,
    "plug_in_motor_vehicle": 0.0,
    "alt_motor_vehicle": 0.0,
}

_SCHED_C_DEFAULTS: Dict[str, Any] = {
    "gross_receipts": 0.0,
    "returns_and_allowances": 0.0,
    "cost_of_goods_sold": 0.0,
    "other_inc_sched_c": 0.0,
    "total_expenses": 0.0,
    "expenses_of_home": 0.0,
    "total_social_security_wages": 0.0,
}
_SCHED_A_DEFAULTS: Dict[str, Any] = {
    "medical_dental_expenses": 0.0,
    "state_local_income_or_sales_tax": 0.0,
    "state_local_real_estate_tax": 0.0,
    "state_local_personal_property_tax": 0.0,
    "other_taxes_paid": 0.0,
    "home_mortgage_interest_and_points": 0.0,
    "home_mortgage_interest_unreported": 0.0,
    "home_mortgage_points_unreported": 0.0,
    "investment_interest": 0.0,
    "charity_cash": 0.0,
    "charity_non_cash": 0.0,
    "casualty_and_theft_loss": 0.0,
    "other_itemized_deductions": 0.0,
}
_EDU_DEFAULTS: Dict[str, Any] = {
    "student_list": [],
}


class L3_ReAct_Experiment(BaseExperiment):
    def __init__(self):
        super().__init__(
            experiment_name="l3_react",
            model_id=MODEL_CONFIG["model_id"],
            )
        self.debug = MODEL_CONFIG.get("debug", False)
    
    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        start_time = time.time()
        acc = get_token_accumulator()
        input_before = acc.total_prompt_tokens
        output_before = acc.total_completion_tokens
        try:
            r = run_l3_instance(
                problem=instance,
                domain=instance.domain,
                model=MODEL_CONFIG["model_id"],
                debug=self.debug,
            )
            predicted = r["answer"]
            # coerce the answer for NBA domain before comparing
            if instance.domain == "nba" and isinstance(predicted, str):
                if predicted.strip().lower() == "true":
                    predicted = True
                elif predicted.strip().lower() == "false":
                    predicted = False

            exact, tolerance = self.compare_answers(predicted, instance.ground_truth_answer)
            input_tokens = acc.total_prompt_tokens - input_before
            output_tokens = acc.total_completion_tokens - output_before
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=predicted,
                expected=instance.ground_truth_answer,
                is_correct_exact=exact,
                is_correct_tolerance=tolerance,
                latency_ms=r["total_time_ms"],
                cost_usd=calculate_cost(input_tokens, output_tokens),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                calculator_name=instance.domain,
                failure_mode="none" if exact else "reasoning_error",
                metadata={
                    "domain": instance.domain,
                    "total_llm_calls": r["total_llm_calls"],
                    "termination_reason": r["termination_reason"],
                    "trajectory": r["trajectory"],
                },
            )

        except Exception as e:
            input_tokens = acc.total_prompt_tokens - input_before
            output_tokens = acc.total_completion_tokens - output_before
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=calculate_cost(input_tokens, output_tokens),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                calculator_name=instance.domain,
                error=str(e),
                failure_mode="agent_error",
                metadata={"domain": instance.domain},
            )

                     
# ============================================================================
# Python calculator wrappers with agent-friendly single-dict interface
# ============================================================================

def compute_airline_fee_ptool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute airline baggage fee and total ticket cost.

    Pass the dict returned by the extraction tool directly as params.

    Required keys in params:
      base_price     - integer ticket price in USD
      customer_class - "Basic Economy" | "Main Cabin" | "Main Plus" |
                       "Premium Economy" | "Business" | "First"
      routine        - destination region, e.g. "U.S.", "Japan", "Europe"
      direction      - 0 = departing from US, 1 = arriving to US
      bag_list       - list of bag dicts (id, name, size, weight)

    Returns:
      {"result": <total cost as int>, "calculator": "airline_baggage_fee"}
      or {"result": None, "error": "<message>", "calculator": "airline_baggage_fee"}
    """
    try:
        total = _airline_calc_raw(params)
        return {"result": total, "calculator": "airline_baggage_fee"}
    except Exception as exc:
        return {"result": None, "error": str(exc), "calculator": "airline_baggage_fee"}


def compute_tax_fee_ptool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute federal tax amount from extracted TaxPayer fields.

    Pass the dict returned by the extraction tool directly as params.

    params should contain the TaxPayer fields produced by extract_tax_params
    (e.g. filing_status, wage_tip_compensation, age, ...).  Optional schedule
    fields (Schedule A, Schedule C, Form 8863) are defaulted to 0 / empty
    when absent — do not attempt to supply them yourself.

    Returns:
      {"result": <amount as float>, "calculator": "tax_fee"}
      Positive = amount owed. Negative = overpaid (refund).
      or {"result": None, "error": "<message>", "calculator": "tax_fee"}

    """
    pydantic_dict = dict(params)
    for defaults in (_TAXPAYER_REQUIRED_DEFAULTS, _SCHED_C_DEFAULTS, _SCHED_A_DEFAULTS, _EDU_DEFAULTS):
        for k, v in defaults.items():
            pydantic_dict.setdefault(k, v)

    try:
        result = _tax_calc_raw({"pydantic": pydantic_dict})
        return {"result": result, "calculator": "tax_fee"}
    except Exception as exc:
        return {"result": None, "error": str(exc), "calculator": "tax_fee"}


# ============================================================================
# PToolSpec wrappers
# model="python" → ReActAgent._execute_action calls spec.func(**args) directly
# (see react.py line 772: if spec.model == "python" and spec.func is not None)
# ============================================================================

AIRLINE_CALC_SPEC = PToolSpec(
    name="compute_airline_fee",
    func=compute_airline_fee_ptool,
    docstring=compute_airline_fee_ptool.__doc__ or "",
    parameters={"params": Dict[str, Any]},
    return_type=Dict[str, Any],
    model="python",
    output_mode="structured",
)

TAX_CALC_SPEC = PToolSpec(
    name="compute_tax_fee",
    func=compute_tax_fee_ptool,
    docstring=compute_tax_fee_ptool.__doc__ or "",
    parameters={"params": Dict[str, Any]},
    return_type=Dict[str, Any],
    model="python",
    output_mode="structured",
)


# ============================================================================
# get_rulearena_ptools(domain)
# ============================================================================

def get_rulearena_ptools(domain: str) -> List[PToolSpec]:
    """
    Return the ReActAgent tool list for the given RuleArena domain.

    Importing this module triggers l1_ptool import which registers the
    extraction ptools in the global registry.  We access them via .spec
    on the PToolWrapper objects, which is more robust than registry lookup
    (the registry keys are private names like _extract_airline_params_fn).

    Args:
        domain: "airline", "tax", or "nba"

    Returns:
        List[PToolSpec] ready to pass as available_ptools= to ReActAgent.

    Tool lists:
        airline: [extract_airline_params.spec, AIRLINE_CALC_SPEC]
        tax:     [extract_tax_params.spec,     TAX_CALC_SPEC]
        nba:     [extract_nba_params.spec]      (no calculator; verdict is direct)
    """
    if domain == "airline":
        return [extract_airline_params.spec, AIRLINE_CALC_SPEC]
    elif domain == "tax":
        return [extract_tax_params.spec, TAX_CALC_SPEC]
    elif domain == "nba":
        return [extract_nba_params.spec]
    else:
        raise ValueError(f"Unknown domain: {domain!r}. Expected 'airline', 'tax', or 'nba'.")


# ============================================================================
# Goal string builders (one per domain)
#
# These describe the task in natural language.  The ReActAgent's thought
# prompt already lists available tools with their signatures, so goal strings
# do NOT name specific tools — that avoids desync if tool names change.
# ============================================================================

def _build_airline_goal(instance: RuleArenaInstance) -> str:
    rules = instance.rules_text[:3000]
    return (
        f"RULES:\n{rules}\n\n"
        f"QUERY:\n{instance.problem_text}\n\n"
        "Your task: compute the total cost (ticket price + all baggage fees).\n"
        "Step 1: Use the extraction tool to parse the query into structured parameters.\n"
        "Step 2: Pass those parameters to the calculator tool to get the total cost.\n"
        "Final answer must be a plain integer (dollars, no $ symbol).\n"
        "Output it as: <answer>INTEGER</answer>"
    )


def _build_tax_goal(instance: RuleArenaInstance) -> str:
    forms_text = build_tax_query(instance.metadata)
    goal = (
        f"{forms_text}\n\n"
        "Your task: compute the federal tax amount owed (or refund if negative).\n"
        "Step 1: Call extract_tax_params ONCE with the complete form text above as the query.\n"
        "         The tool will return all TaxPayer fields - do NOT call it again regardless of what the result looks like.\n"
        "Step 2: Call compute_tax_fee with EXACTLY this argument: params=$extract_tax_params\n"
        "         This passes the full extraction result by reference - do NOT reconstruct or retype the params dict.\n"
        "Final answer is a decimal number. Positive = owed. Negative = refund.\n"
        "Output it as: <answer>NUMBER</answer>"
    )
    # Sanitize non-ASCII characters from the form text (e.g. em-dashes, copyright symbols)
    # that crash Windows cp1252 terminals. The LLM receives the original via the ptool query.
    return goal.encode("ascii", errors="replace").decode("ascii")


def _build_nba_goal(instance: RuleArenaInstance) -> str:
    query = _build_nba_query(instance)
    return (
        f"{query}\n\n"
        f"------------------------------------"
        f"------------------------------------"
        "Your task: determine whether any team operation violates NBA CBA rules.\n"
        "Step 1: Call the extraction tool once with the full query above.\n"
        "Step 2: The tool returns verdict (True/False), illegal_operation, "
        "problematic_team, and reasoning.\n"
        "The verdict field IS your final answer. Do not call the tool again.\n"
        "Immediately output: <answer>True</answer> or <answer>False</answer>"
    )


_GOAL_BUILDERS = {
    "airline": _build_airline_goal,
    "tax": _build_tax_goal,
    "nba": _build_nba_goal,
}


# ============================================================================
# run_l3_instance
# ============================================================================

def run_l3_instance(
    problem: RuleArenaInstance,
    domain: str,
    model: str,
    max_steps: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the ReActAgent on a single RuleArena instance.

    Args:
        problem:    A RuleArenaInstance from RuleArenaDataset.
        domain:     "airline", "tax", or "nba".
        model:      LLM model ID for the ReAct reasoning loop.
        max_steps:  Maximum think-act-observe iterations (default 10).

    Returns:
        {
            "answer":             str | None   — content of <answer> tag, or None
            "success":            bool         — True if <answer> tag was found
            "trajectory":         dict         — full ReActTrajectory.to_dict()
            "termination_reason": str          — "answer_found"|"max_steps"|"error"
            "total_llm_calls":    int
            "total_time_ms":      float
        }

    Notes:
        store_trajectories=False suppresses disk I/O to ~/.react_traces/.
        The full trajectory is available in the returned dict.

        The agent stores each tool result under context[tool_name] and
        context[f"step_{n}"].  The $ref resolution in _resolve_args lets
        the agent pass "$<tool_name>" to reference a prior result by name.
    """
    if domain not in _GOAL_BUILDERS:
        raise ValueError(f"Unknown domain: {domain!r}. Expected airline, tax, or nba.")

    goal = _GOAL_BUILDERS[domain](problem)
    ptools = get_rulearena_ptools(domain)

    agent = ReActAgent(
        available_ptools=ptools,
        model=model,
        max_steps=max_steps,
        echo=debug,
        store_trajectories=True,
    )

    result = agent.run(goal)

    return {
        "answer": result.answer,
        "success": result.success,
        "trajectory": result.trajectory.to_dict(),
        "termination_reason": result.trajectory.termination_reason,
        "total_llm_calls": result.trajectory.total_llm_calls,
        "total_time_ms": result.trajectory.total_time_ms,
    }
