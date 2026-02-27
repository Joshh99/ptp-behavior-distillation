"""
L1: PTool Extraction + Python Calculation

Two-phase approach using the @ptool decorator:
1. LLM extracts structured parameters as JSON (@ptool)
2. Python deterministic calculator computes answer

This is the core PTP hypothesis test: separation of extraction (LLM) from
computation (Python) improves reliability over end-to-end LLM reasoning.

Supported domains:
- airline: extract baggage params -> compute_airline_fee()
- tax: extract TaxPayer fields from filled IRS forms -> compute_tax_fee()
- nba: extract boolean verdict directly (no calculator)
"""

import sys
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# ptool framework lives under external/
REPO_ROOT = Path(__file__).parent.parent.parent.parent
_EXTERNAL = str(REPO_ROOT / "external")
if _EXTERNAL not in sys.path:
    sys.path.insert(0, _EXTERNAL)

from ptool_framework.ptool import ptool, PToolExample
from ptool_framework.llm_backend import call_llm_compat as call_llm, parse_structured_response, ParseError

from benchmark.rulearena.dataset.loader import RuleArenaInstance
from benchmark.rulearena.experiments.base import BaseExperiment, ExperimentResult
from benchmark.rulearena.config import MODEL_CONFIG, calculate_cost

# Model name as registered in LLMS.json (must match a key there, not an alias)
_MODEL = MODEL_CONFIG["model_id"]  # "deepseek-ai/DeepSeek-V3"


# ============================================================================
# @ptool definitions — one per domain
# ============================================================================

# --- Airline ---

_AIRLINE_DOC = """\
Extract structured baggage parameters from an airline baggage-fee query.

Return a JSON object with these exact fields:
{
    "base_price": <integer ticket price in USD>,
    "customer_class": <one of: "Basic Economy", "Main Cabin", "Main Plus",
                       "Premium Economy", "Business", "First">,
    "routine": <destination region, e.g. "U.S.", "Japan", "Europe">,
    "direction": <0 for departing from US, 1 for arriving to US>,
    "bag_list": [
        {"id": 1, "name": "<item>", "size": [length, width, height], "weight": <lbs>},
        ...
    ]
}

Requirements:
- customer_class must be an exact match from the list above
- routine must match the rules (e.g. "U.S." with period for domestic)
- direction: 0=from US, 1=to US
- size: array of 3 integers [length, width, height] in inches
- weight: integer in pounds
"""

_AIRLINE_EXAMPLE = PToolExample(
    inputs={"query": (
        "A Main Cabin passenger is flying from Chicago to Tokyo with a $1200 ticket. "
        "They have a carry-on bag (22x14x9 inches, 15 lbs) and a checked suitcase "
        "(28x20x12 inches, 45 lbs)."
    )},
    output={
        "base_price": 1200,
        "customer_class": "Main Cabin",
        "routine": "Japan",
        "direction": 0,
        "bag_list": [
            {"id": 1, "name": "carry-on bag", "size": [22, 14, 9], "weight": 15},
            {"id": 2, "name": "checked suitcase", "size": [28, 20, 12], "weight": 45},
        ],
    },
)


def _extract_airline_params_fn(query: str) -> dict:
    ...


_extract_airline_params_fn.__doc__ = _AIRLINE_DOC
extract_airline_params = ptool(
    model=_MODEL,
    output_mode="structured",
    examples=[_AIRLINE_EXAMPLE],
)(_extract_airline_params_fn)


# --- Tax ---

_TAX_DOC = """\
Extract taxpayer parameters from filled IRS forms.

The input is a set of IRS forms with dollar values already filled in and
computed fields marked [__]. Extract the INPUT values that appear on the forms.

Return a JSON object with these fields:
{
    // --- Basic info ---
    "name": "<taxpayer name>",
    "age": <int>,
    "spouse_age": <int>,
    "filing_status": "<single|married filing jointly|married filing separately|head of household|qualifying surviving spouse>",
    "blind": <bool>,
    "spouse_blind": <bool>,
    "itemized": <bool>,
    "self_employed": <bool>,
    "has_student_loans_or_education_expenses": <bool>,
    "num_qualifying_children": <int>,
    "num_other_dependents": <int>,

    // --- Form 1040 income lines ---
    "wage_tip_compensation": <float>,
    "household_employee_wage": <float>,
    "unreported_tip": <float>,
    "nontaxable_combat_pay": <float>,
    "tax_exempt_interest": <float>,
    "taxable_interest": <float>,
    "qualified_dividends": <float>,
    "ordinary_dividends": <float>,
    "ira_distributions": <float>,
    "taxable_ira_distributions": <float>,
    "all_pensions": <float>,
    "taxable_pensions": <float>,
    "social_security_benefits": <float>,
    "taxable_social_security_benefits": <float>,
    "qualified_business_income": <float>,  // Line 13. Read 'Deduction from Form 8995 or Form 8995-A' value. Do NOT read [__] as zero.
    "federal_income_tax_withheld": <float>,
    "earned_income_credit": <float>,

    // --- Schedule 1 ---
    "taxable_state_refunds": <float>,
    "alimony_income": <float>,
    "sale_of_business": <float>,  // Line 4 'Other gains or (losses)'
    "rental_real_estate_sch1": <float>,
    "farm_income": <float>,
    "unemployment_compensation": <float>,
    "other_income": <float>,
    "educator_expenses": <float>,
    "hsa_deduction": <float>,
    "ira_deduction": <float>,
    "student_loan_interest_deduction": <float>,
    "other_adjustments": <float>,

    // --- Schedule 2 ---
    "amt_f6251": <float>,
    "credit_repayment": <float>,
    "other_additional_taxes": <float>,

    // --- Schedule 3 ---
    "foreign_tax_credit": <float>,
    "dependent_care": <float>,
    "retirement_savings": <float>,
    "elderly_disabled_credits": <float>,
    "plug_in_motor_vehicle": <float>,
    "alt_motor_vehicle": <float>,

    // --- Schedule A (only when itemized=true) ---
    "medical_dental_expenses": <float>,            // Line 1
    "state_local_income_or_sales_tax": <float>,    // Line 5a
    "state_local_real_estate_tax": <float>,         // Line 5b
    "state_local_personal_property_tax": <float>,   // Line 5c
    "other_taxes_paid": <float>,                    // Line 6
    "home_mortgage_interest_and_points": <float>,   // Line 8a
    "home_mortgage_interest_unreported": <float>,   // Line 8b
    "home_mortgage_points_unreported": <float>,     // Line 8c
    "investment_interest": <float>,                 // Line 9
    "charity_cash": <float>,                        // Line 11
    "charity_non_cash": <float>,                    // Line 12
    "casualty_and_theft_loss": <float>,             // Line 15
    "other_itemized_deductions": <float>,           // Line 16

    // --- Schedule C (only when self_employed=true) ---
    "gross_receipts": <float>,            // Line 1
    "returns_and_allowances": <float>,    // Line 2
    "cost_of_goods_sold": <float>,        // Line 4
    "other_inc_sched_c": <float>,         // Line 6
    "total_expenses": <float>,            // Line 28
    "expenses_of_home": <float>,          // Line 30
    "total_social_security_wages": <float>, // Schedule SE Line 8a (or W-2 boxes 3+7)

    // --- Form 8863 education (only when has_student_loans_or_education_expenses=true) ---
    "student_list": [   // One entry per student in Part III
        {
            "qualified_student_expenses": <int>,    // Part III Line 21
            "f8863_part_iii_23": "<Yes or No>",     // Line 23
            "f8863_part_iii_24": "<Yes or No>",     // Line 24
            "f8863_part_iii_25": "<Yes or No>",     // Line 25
            "f8863_part_iii_26": "<Yes or No>"      // Line 26
        }
    ]
}

Extraction rules:
- Dollar values on forms appear as "$1,234" — extract as numeric (1234.0).
- Fields for schedules not present in the forms should be 0, 0.0, or [].
- self_employed = true if Schedule C is present.
- has_student_loans_or_education_expenses = true if Form 8863 is present.
- itemized = true if Schedule A is present.
- student_list = [] if Form 8863 is not present.
"""

_TAX_EXAMPLE = PToolExample(
    inputs={"query": "(abbreviated) Name: Jane, Age: 35, Filing Status: single, "
            "Line 1a W-2: $50,000, Line 2b Taxable interest: $200 ..."},
    output={
        "name": "Jane",
        "age": 35,
        "spouse_age": 0,
        "filing_status": "single",
        "blind": False,
        "spouse_blind": False,
        "itemized": False,
        "self_employed": False,
        "has_student_loans_or_education_expenses": False,
        "num_qualifying_children": 0,
        "num_other_dependents": 0,
        "wage_tip_compensation": 50000.0,
        "household_employee_wage": 0.0,
        "unreported_tip": 0.0,
        "nontaxable_combat_pay": 0.0,
        "tax_exempt_interest": 0.0,
        "taxable_interest": 200.0,
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
        "amt_f6251": 0.0,
        "credit_repayment": 0.0,
        "other_additional_taxes": 0.0,
        "foreign_tax_credit": 0.0,
        "dependent_care": 0.0,
        "retirement_savings": 0.0,
        "elderly_disabled_credits": 0.0,
        "plug_in_motor_vehicle": 0.0,
        "alt_motor_vehicle": 0.0,
        # Schedule A (not present for this taxpayer)
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
        # Schedule C (not present for this taxpayer)
        "gross_receipts": 0.0,
        "returns_and_allowances": 0.0,
        "cost_of_goods_sold": 0.0,
        "other_inc_sched_c": 0.0,
        "total_expenses": 0.0,
        "expenses_of_home": 0.0,
        "total_social_security_wages": 0.0,
        # Form 8863 (not present for this taxpayer)
        "student_list": [],
    },
)


def _extract_tax_params_fn(query: str) -> dict:
    ...


_extract_tax_params_fn.__doc__ = _TAX_DOC
extract_tax_params = ptool(
    model=_MODEL,
    output_mode="structured",
    examples=[_TAX_EXAMPLE],
)(_extract_tax_params_fn)


# --- NBA ---

_NBA_DOC = """\
Determine whether any NBA team operation violates the Collective Bargaining
Agreement salary cap rules.

You are given:
- Reference rules from the NBA CBA
- Team and player salary situations
- A list of proposed operations (signings, trades, etc.)

Analyze each operation against the CBA rules. Consider salary cap exceptions,
traded player exceptions, sign-and-trade rules, maximum salaries, etc.

Return a JSON object:
{
    "verdict": <true if ANY operation violates the rules, false if all are compliant>,
    "illegal_operation": "<letter A/B/C/... of the violating operation, or empty string>",
    "problematic_team": "<letter A/B/... of the team that violates, or empty string>",
    "reasoning": "<brief explanation>"
}
"""

_NBA_EXAMPLE = PToolExample(
    inputs={"query": (
        "Rules: [salary cap is $140,588,000 ...]\n\n"
        "Team Situations:\n"
        "Team A has team salary $130,000,000 ...\n\n"
        "Player Situations:\n"
        "Player A is a 5-year veteran ...\n\n"
        "Operations:\n"
        "A. Team A signs Player A to a 3-year contract at $15,000,000/year."
    )},
    output={
        "verdict": False,
        "illegal_operation": "",
        "problematic_team": "",
        "reasoning": "Team A's salary after signing ($145M) is within "
                     "the cap with standard exceptions.",
    },
)


def _extract_nba_params_fn(query: str) -> dict:
    ...


_extract_nba_params_fn.__doc__ = _NBA_DOC
extract_nba_params = ptool(
    model=_MODEL,
    output_mode="structured",
    examples=[_NBA_EXAMPLE],
)(_extract_nba_params_fn)


# ============================================================================
# Region normalization (airline)
# ============================================================================

VALID_REGIONS = {
    "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama",
    "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar",
    "Europe", "India", "China", "Japan", "South Korea", "Hong Kong",
    "Australia", "New Zealand",
}

REGION_FIXES = {
    "asia": "China",
    "north america": "U.S.",
    "us": "U.S.",
    "usa": "U.S.",
    "united states": "U.S.",
    "domestic": "U.S.",
    "tokyo": "Japan",
    "beijing": "China",
    "shanghai": "China",
    "seoul": "South Korea",
    "sydney": "Australia",
    "london": "Europe",
    "paris": "Europe",
    "berlin": "Europe",
}


def _normalize_region(routine: str) -> str:
    if routine in VALID_REGIONS:
        return routine
    fixed = REGION_FIXES.get(routine.lower().strip())
    return fixed if fixed else "U.S."


# ============================================================================
# Query builders (reuse tax builder from l0f_cot, build NBA query here)
# ============================================================================

def _build_nba_query(instance: RuleArenaInstance) -> str:
    """Build NBA query from instance metadata, matching auto_test.py format."""
    md = instance.metadata
    team_info = "Team Situations:\n" + "\n".join(md.get("team_situations", []))
    player_info = "Player Situations:\n" + "\n".join(md.get("player_situations", []))
    operations = "Operations:\n" + "\n".join(md.get("operations", []))
    question = team_info + "\n\n" + player_info + "\n\n" + operations

    assumptions = (
        "Assume:\n"
        "* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;\n"
        "* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;\n"
        "* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;\n"
        "* the Luxury Tax is $170,814,000;\n"
        "* the First Apron Level is $178,132,000;\n"
        "* the Second Apron Level is $188,931,000;\n"
        "* the Team Salary of each team listed under \"Team Situations:\" do not "
        "include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.\n"
    )

    rules_text = instance.rules_text[:6000]

    return (
        f"Reference Rules in NBA Collective Bargaining Agreement:\n\n"
        f"{rules_text}\n\n"
        f"{assumptions}\n"
        f"Decide whether any operation by any team violates the rules:\n\n"
        f"{question}"
    )


# ============================================================================
# NBA fallback parser
# ============================================================================

def _extract_nba_verdict_fallback(raw: str) -> dict:
    """Regex extraction of NBA verdict fields from malformed JSON.

    Used when the full JSON parser fails (e.g. unescaped quotes or
    truncation in the reasoning field).  The three fields we need
    always appear before reasoning in the LLM output.
    """
    result = {}
    m = re.search(r'"verdict"\s*:\s*(true|false)', raw, re.IGNORECASE)
    result["verdict"] = m.group(1).lower() == "true" if m else False

    m = re.search(r'"illegal_operation"\s*:\s*"([^"]*)"', raw)
    if m:
        result["illegal_operation"] = m.group(1)

    m = re.search(r'"problematic_team"\s*:\s*"([^"]*)"', raw)
    if m:
        result["problematic_team"] = m.group(1)

    result["reasoning"] = "(extracted via fallback — original JSON malformed)"
    return result


# ============================================================================
# Experiment class
# ============================================================================

class L1_PTool_Experiment(BaseExperiment):
    """
    L1: PTool extraction experiment using @ptool decorator.

    Phase 1: @ptool extracts structured parameters via LLM
    Phase 2: Python calculator computes answer (airline, tax) or
             verdict returned directly (NBA)
    """

    def __init__(self):
        super().__init__(
            experiment_name="l1_ptool",
            model_id=MODEL_CONFIG["model_id"],
        )

    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        start_time = time.time()

        try:
            if instance.domain == "airline":
                return self._run_airline(instance, start_time)
            elif instance.domain == "tax":
                return self._run_tax(instance, start_time)
            elif instance.domain == "nba":
                return self._run_nba(instance, start_time)
            else:
                raise ValueError(f"Unknown domain: {instance.domain}")

        except ParseError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                error=str(e),
                failure_mode="extraction_failure",
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                },
            )
        except (KeyError, AttributeError) as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                error=str(e),
                failure_mode="scope_error",
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                },
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                error=str(e),
                failure_mode="calculation_error",
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                },
            )

    # ------------------------------------------------------------------
    # Airline
    # ------------------------------------------------------------------
    def _run_airline(self, instance, start_time):
        query = f"RULES:\n{instance.rules_text[:3000]}\n\nQUERY:\n{instance.problem_text}"
        prompt = extract_airline_params.spec.format_prompt(query=query)
        raw_response, in_tok, out_tok = call_llm(prompt, _MODEL)
        params = parse_structured_response(raw_response, dict)

        # Normalize region
        params["routine"] = _normalize_region(params.get("routine", "U.S."))

        from benchmark.rulearena.calculators.airline import compute_airline_fee
        predicted = compute_airline_fee(params)

        return self._build_result(
            instance, start_time, predicted, raw_response, in_tok, out_tok,
            extra_meta={"extraction_json": params},
        )

    # ------------------------------------------------------------------
    # Tax
    # ------------------------------------------------------------------

    # Fields on Form1040 (not TaxPayer) that compute_answer() accesses
    # via TaxPayer's extra="allow".  Grouped by the boolean guard that
    # gates their access inside compute_answer / helper functions.
    _SCHED_C_DEFAULTS = {
        "gross_receipts": 0.0,
        "returns_and_allowances": 0.0,
        "cost_of_goods_sold": 0.0,
        "other_inc_sched_c": 0.0,
        "total_expenses": 0.0,
        "expenses_of_home": 0.0,
        "total_social_security_wages": 0.0,
    }
    _SCHED_A_DEFAULTS = {
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
    _EDU_DEFAULTS = {
        "student_list": [],
    }

    def _run_tax(self, instance, start_time):
        # Build the filled IRS forms query (same as L0F)
        from benchmark.rulearena.experiments.l0f_cot import build_tax_query
        forms_query = build_tax_query(instance.metadata)

        prompt = extract_tax_params.spec.format_prompt(query=forms_query)
        raw_response, in_tok, out_tok = call_llm(prompt, _MODEL)
        params = parse_structured_response(raw_response, dict)

        # Build a pseudo problem_data dict that compute_tax_fee expects
        pydantic_dict = dict(params)  # shallow copy

        # Supply defaults for Form1040-only fields that compute_answer()
        # accesses via TaxPayer(extra="allow").  Without these, attribute
        # lookups crash for self-employed / itemized / education problems.
        for defaults in (self._SCHED_C_DEFAULTS, self._SCHED_A_DEFAULTS, self._EDU_DEFAULTS):
            for k, v in defaults.items():
                pydantic_dict.setdefault(k, v)

        tax_info = {"pydantic": pydantic_dict}

        from benchmark.rulearena.calculators.tax import compute_tax_fee
        predicted = compute_tax_fee(tax_info)
        if predicted is None:
            raise RuntimeError("Tax calculator returned None")

        return self._build_result(
            instance, start_time, predicted, raw_response, in_tok, out_tok,
            extra_meta={"extraction_json": params},
        )

    # ------------------------------------------------------------------
    # NBA
    # ------------------------------------------------------------------
    def _run_nba(self, instance, start_time):
        query = _build_nba_query(instance)
        prompt = extract_nba_params.spec.format_prompt(query=query)
        raw_response, in_tok, out_tok = call_llm(prompt, _MODEL)

        try:
            params = parse_structured_response(raw_response, dict)
        except ParseError:
            # Last resort: regex extraction when JSON is unrecoverable.
            # verdict/illegal_operation/problematic_team appear before the
            # long reasoning field, so simple regexes reliably capture them.
            params = _extract_nba_verdict_fallback(raw_response)

        # NBA returns boolean verdict directly — no calculator
        predicted = bool(params.get("verdict", False))

        return self._build_result(
            instance, start_time, predicted, raw_response, in_tok, out_tok,
            extra_meta={"extraction_json": params},
        )

    # ------------------------------------------------------------------
    # Shared result builder
    # ------------------------------------------------------------------
    def _build_result(
        self,
        instance: RuleArenaInstance,
        start_time: float,
        predicted: Any,
        raw_response: str,
        input_tokens: int,
        output_tokens: int,
        calc_error: Optional[str] = None,
        extra_meta: Optional[Dict] = None,
    ) -> ExperimentResult:
        exact, tolerance = self.compare_answers(predicted, instance.ground_truth_answer)
        elapsed_ms = (time.time() - start_time) * 1000
        cost = calculate_cost(input_tokens, output_tokens)

        # Determine failure mode
        if predicted is None:
            failure_mode = "extraction_failure"
        elif exact:
            failure_mode = "none"
        else:
            failure_mode = "calculation_error"

        meta = {
            "domain": instance.domain,
            "complexity_level": instance.complexity_level,
        }
        if extra_meta:
            meta.update(extra_meta)

        return ExperimentResult(
            instance_id=instance.instance_id,
            predicted=predicted,
            expected=instance.ground_truth_answer,
            is_correct_exact=exact,
            is_correct_tolerance=tolerance,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            calculator_name=instance.domain,
            error=calc_error,
            failure_mode=failure_mode,
            raw_response=raw_response,
            metadata=meta,
        )


# ============================================================================
# Standalone smoke test
# ============================================================================

if __name__ == "__main__":
    from benchmark.rulearena.dataset.loader import RuleArenaDataset

    print("Testing L1 PTool Experiment...")
    dataset = RuleArenaDataset()

    # Get one airline instance
    airline_instances = [inst for inst in dataset.instances if inst.domain == "airline"]
    sample = airline_instances[:1]

    experiment = L1_PTool_Experiment()
    result = experiment.run_instance(sample[0])

    print(f"\nResult:")
    print(f"  Instance: {result.instance_id}")
    print(f"  Predicted: {result.predicted}")
    print(f"  Expected: {result.expected}")
    print(f"  Correct: {result.is_correct_exact}")
    print(f"  Cost: ${result.cost_usd:.4f}")
    print(f"  Time: {result.latency_ms:.0f}ms")
    if result.error:
        print(f"  Error: {result.error}")
    if result.metadata.get("extraction_json"):
        print(f"  Extracted params: {result.metadata['extraction_json']}")
