"""
Diagnostic: Run actual LLM extraction on tax problems, compare to ground truth.
Identifies which specific field values the LLM gets wrong.

Run from: ptp-behavior-distillation/
Delete after use.
"""

import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — mirrors l1_ptool.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
EXTERNAL = str(REPO_ROOT / "external")
if EXTERNAL not in sys.path:
    sys.path.insert(0, EXTERNAL)

TAX_DIR = str(REPO_ROOT / "external" / "RuleArena" / "tax")
if TAX_DIR not in sys.path:
    sys.path.insert(0, TAX_DIR)

# Import ptool framework (same path l1_ptool.py uses)
from ptool_framework.llm_backend import call_llm, parse_structured_response, get_config

# Import build_tax_query (same function _run_tax calls)
from benchmark.rulearena.experiments.l0f_cot import build_tax_query

# Import the tax ptool spec (need to trigger l1_ptool module load to register it)
from benchmark.rulearena.experiments.l1_ptool import extract_tax_params
from benchmark.rulearena.config import MODEL_CONFIG

_MODEL = MODEL_CONFIG["model_id"]

# Import calculator for answer verification
from structured_forms import TaxPayer
from micro_evaluation import compute_answer

# ---------------------------------------------------------------------------
# L1 extraction schema fields (copied from _TAX_DOC in l1_ptool.py)
# ---------------------------------------------------------------------------
L1_SCHEMA_FIELDS = [
    "name", "age", "spouse_age", "filing_status", "blind", "spouse_blind",
    "itemized", "self_employed", "has_student_loans_or_education_expenses",
    "num_qualifying_children", "num_other_dependents",
    "wage_tip_compensation", "household_employee_wage", "unreported_tip",
    "nontaxable_combat_pay", "tax_exempt_interest", "taxable_interest",
    "qualified_dividends", "ordinary_dividends", "ira_distributions",
    "taxable_ira_distributions", "all_pensions", "taxable_pensions",
    "social_security_benefits", "taxable_social_security_benefits",
    "qualified_business_income", "federal_income_tax_withheld",
    "earned_income_credit", "taxable_state_refunds", "alimony_income",
    "sale_of_business", "rental_real_estate_sch1", "farm_income",
    "unemployment_compensation", "other_income", "educator_expenses",
    "hsa_deduction", "ira_deduction", "student_loan_interest_deduction",
    "other_adjustments", "amt_f6251", "credit_repayment",
    "other_additional_taxes", "foreign_tax_credit", "dependent_care",
    "retirement_savings", "elderly_disabled_credits",
    "plug_in_motor_vehicle", "alt_motor_vehicle",
]

# Numeric fields only (skip string/bool fields for delta computation)
NUMERIC_FIELDS = [f for f in L1_SCHEMA_FIELDS if f not in (
    "name", "filing_status", "blind", "spouse_blind", "itemized",
    "self_employed", "has_student_loans_or_education_expenses",
)]

BOOL_FIELDS = [
    "blind", "spouse_blind", "itemized", "self_employed",
    "has_student_loans_or_education_expenses",
]

INT_FIELDS = ["age", "spouse_age", "num_qualifying_children", "num_other_dependents"]


def find_snippet(query_text, field_name, value):
    """Find a ~120-char snippet from query_text containing the value."""
    if isinstance(value, bool):
        # For bools, search for the field concept
        patterns = [field_name.replace("_", " "), str(value)]
    elif isinstance(value, (int, float)):
        # Search for the numeric value formatted as $X,XXX or plain
        v = int(value) if value == int(value) else value
        patterns = [
            "${:,}".format(v),       # $1,234
            "{:,}".format(v),        # 1,234
            str(v),                  # 1234
        ]
    else:
        patterns = [str(value)]

    for pat in patterns:
        idx = query_text.find(pat)
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(query_text), idx + len(pat) + 80)
            snippet = query_text[start:end].replace("\n", " ")
            return "...{}...".format(snippet)
    return "<NOT FOUND in query text>"


def main():
    # ------------------------------------------------------------------
    # 1. Load comp_0 problems
    # ------------------------------------------------------------------
    data_path = Path(TAX_DIR) / "synthesized_problems" / "comp_0.json"
    with open(data_path) as f:
        all_problems = json.load(f)

    N = min(5, len(all_problems))
    problems = all_problems[:N]
    print("Loaded {} comp_0 problems (using first {})".format(len(all_problems), N))

    # Verify model
    config = get_config()
    model_config = config.get_model(_MODEL)
    print("Model: {} ({})".format(model_config.name, model_config.model_id))
    print("Provider: {}".format(model_config.provider))
    print()

    total_wrong_fields = {}  # field_name -> count of problems where it was wrong

    for pi in range(N):
        p = problems[pi]
        pyd = p["pydantic"]

        print("=" * 78)
        print("PROBLEM {} / {}".format(pi, N - 1))
        print("=" * 78)

        # ------------------------------------------------------------------
        # 2. Build query (same path as _run_tax)
        # ------------------------------------------------------------------
        forms_query = build_tax_query(p)

        print("\n--- Query text (first 500 chars) ---")
        print(forms_query[:500])
        print("... [{} total chars]".format(len(forms_query)))

        # ------------------------------------------------------------------
        # 3. Ground truth (only L1 schema fields)
        # ------------------------------------------------------------------
        gt = {}
        for f in L1_SCHEMA_FIELDS:
            if f in pyd:
                gt[f] = pyd[f]

        print("\n--- Ground truth (L1 schema fields) ---")
        for f in L1_SCHEMA_FIELDS:
            if f in gt:
                print("  {:45s} = {}".format(f, gt[f]))

        # ------------------------------------------------------------------
        # 4. Run actual LLM extraction (same path as _run_tax)
        # ------------------------------------------------------------------
        print("\n--- Calling LLM extraction... ---")
        t0 = time.time()
        prompt = extract_tax_params.spec.format_prompt(query=forms_query)
        raw_response, in_tok, out_tok = call_llm(prompt, _MODEL)
        elapsed = time.time() - t0
        print("  Done in {:.1f}s  (in={}, out={} tokens)".format(elapsed, in_tok, out_tok))

        extracted = parse_structured_response(raw_response, dict)

        # ------------------------------------------------------------------
        # 5. Per-field comparison
        # ------------------------------------------------------------------
        print("\n--- Per-field deltas (nonzero only) ---")
        wrong_fields = []

        # Check string/enum fields
        for f in ["name", "filing_status"]:
            gt_val = gt.get(f, "")
            ex_val = extracted.get(f, "")
            if str(gt_val).lower().strip() != str(ex_val).lower().strip():
                wrong_fields.append((f, gt_val, ex_val, None))
                print("  {:45s}  GT={!r:20s}  EX={!r}".format(f, str(gt_val), str(ex_val)))

        # Check bool fields
        for f in BOOL_FIELDS:
            gt_val = gt.get(f)
            ex_val = extracted.get(f)
            if gt_val is None:
                continue
            # Normalize to bool
            if isinstance(ex_val, str):
                ex_bool = ex_val.lower() in ("true", "yes", "1")
            else:
                ex_bool = bool(ex_val) if ex_val is not None else False
            if bool(gt_val) != ex_bool:
                wrong_fields.append((f, gt_val, ex_val, None))
                print("  {:45s}  GT={!r:20s}  EX={!r}".format(f, gt_val, ex_val))

        # Check int fields
        for f in INT_FIELDS:
            gt_val = gt.get(f, 0)
            ex_val = extracted.get(f, 0)
            try:
                delta = int(ex_val) - int(gt_val)
            except (TypeError, ValueError):
                delta = None
            if delta is None or delta != 0:
                wrong_fields.append((f, gt_val, ex_val, delta))
                print("  {:45s}  GT={:<12}  EX={:<12}  delta={}".format(
                    f, gt_val, ex_val, delta))

        # Check numeric (float) fields
        for f in NUMERIC_FIELDS:
            if f in INT_FIELDS:
                continue  # already handled
            gt_val = gt.get(f, 0.0)
            ex_val = extracted.get(f, 0.0)
            try:
                gt_num = float(gt_val)
                ex_num = float(ex_val)
                delta = ex_num - gt_num
            except (TypeError, ValueError):
                gt_num = gt_val
                ex_num = ex_val
                delta = None

            if delta is None or abs(delta) > 0.01:
                wrong_fields.append((f, gt_val, ex_val, delta))
                print("  {:45s}  GT={:<12}  EX={:<12}  delta={}".format(
                    f, gt_num, ex_num,
                    "{:.2f}".format(delta) if delta is not None else "PARSE_ERR"))

        if not wrong_fields:
            print("  (all fields match!)")

        # ------------------------------------------------------------------
        # 6. For wrong fields, show query snippet
        # ------------------------------------------------------------------
        if wrong_fields:
            print("\n--- Query snippets for wrong fields ---")
            for f, gt_val, ex_val, delta in wrong_fields:
                snippet = find_snippet(forms_query, f, gt_val)
                print("  {}:".format(f))
                print("    GT value in query: {}".format(snippet))
                if ex_val is not None:
                    snippet_ex = find_snippet(forms_query, f, ex_val)
                    if snippet_ex != snippet:
                        print("    EX value in query: {}".format(snippet_ex))

        # ------------------------------------------------------------------
        # Answer-level impact
        # ------------------------------------------------------------------
        print("\n--- Answer impact ---")
        # Ground truth answer
        tp_gt = TaxPayer(**pyd)
        gt_answer, _ = compute_answer(tp_gt)

        # Build pydantic dict using LLM-extracted values for L1 fields,
        # plus ground truth for non-L1 fields (to isolate extraction error)
        pyd_with_extraction = dict(pyd)
        for f in L1_SCHEMA_FIELDS:
            if f in extracted:
                pyd_with_extraction[f] = extracted[f]
        try:
            tp_ex = TaxPayer(**pyd_with_extraction)
            ex_answer, _ = compute_answer(tp_ex)
            print("  Ground truth answer:    {}".format(gt_answer))
            print("  With LLM extraction:    {}".format(ex_answer))
            print("  Answer delta:           {}".format(ex_answer - gt_answer))
        except Exception as e:
            print("  Calculator error with extracted values: {}".format(e))

        # Track wrong field frequency
        for f, _, _, _ in wrong_fields:
            total_wrong_fields[f] = total_wrong_fields.get(f, 0) + 1

        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 78)
    print("SUMMARY: Fields wrong across {} problems".format(N))
    print("=" * 78)
    ranked = sorted(total_wrong_fields.items(), key=lambda x: -x[1])
    if ranked:
        for f, count in ranked:
            print("  {:45s}  wrong in {}/{} problems".format(f, count, N))
    else:
        print("  (no field mismatches detected!)")


if __name__ == "__main__":
    main()
