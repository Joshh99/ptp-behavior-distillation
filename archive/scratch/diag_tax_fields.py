"""
Diagnostic: compare L1 extraction schema fields vs ground-truth pydantic fields.
Run from: ptp-behavior-distillation/
Delete after use.
"""

import json
import sys
from pathlib import Path

# RuleArena tax modules use bare imports
TAX_DIR = str(Path(__file__).parent / "external" / "RuleArena" / "tax")
sys.path.insert(0, TAX_DIR)

from structured_forms import TaxPayer
from micro_evaluation import compute_answer

# ---------- L1 extraction schema fields (from l1_ptool.py _TAX_DOC) ----------
L1_FIELDS = {
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
}

# Also list TaxPayer model fields for a three-way comparison
TAXPAYER_FIELDS = set(TaxPayer.model_fields.keys())

# ---------- Load comp_0 dataset ----------
data_path = Path(TAX_DIR) / "synthesized_problems" / "comp_0.json"
with open(data_path) as f:
    problems = json.load(f)

print("=" * 70)
print("THREE-WAY FIELD COMPARISON")
print("=" * 70)

print("\n[A] Fields in TaxPayer model but NOT in L1 schema:")
for k in sorted(TAXPAYER_FIELDS - L1_FIELDS):
    print("    ", k)

print("\n[B] Fields in L1 schema but NOT in TaxPayer model:")
for k in sorted(L1_FIELDS - TAXPAYER_FIELDS):
    print("    ", k)

print("\n[C] Fields in pydantic ground-truth (problem 0) but NOT in L1 schema:")
pyd0_keys = set(problems[0]["pydantic"].keys())
for k in sorted(pyd0_keys - L1_FIELDS):
    print("    ", k)

print("\n[D] Fields in pydantic ground-truth but NOT in TaxPayer model:")
for k in sorted(pyd0_keys - TAXPAYER_FIELDS):
    print("    ", k)

# ---------- Per-problem impact ----------
print("\n" + "=" * 70)
print("PER-PROBLEM IMPACT (comp_0, problems 0-2)")
print("=" * 70)

for i in range(min(3, len(problems))):
    p = problems[i]
    pyd = p["pydantic"]
    pyd_keys = set(pyd.keys())
    missing_from_schema = pyd_keys - L1_FIELDS

    print("\n--- Problem {} ---".format(i))
    print("Filing: {}, Age: {}, Blind: {}".format(
        pyd["filing_status"], pyd["age"], pyd["blind"]
    ))

    # Values of fields present in ground truth but absent from L1 schema
    print("\nMissing-from-L1-schema field values:")
    for k in sorted(missing_from_schema):
        print("    {:40s} = {}".format(k, pyd[k]))

    # Ground truth: construct TaxPayer with ALL pydantic fields
    tp_gt = TaxPayer(**pyd)
    gt_answer, gt_trace = compute_answer(tp_gt)
    print("\nGround truth answer: {}".format(gt_answer))

    # Simulated L1: construct TaxPayer with ONLY L1-schema fields
    # (perfect extraction, but missing fields default to 0/False)
    extracted = {}
    for k in L1_FIELDS:
        if k in pyd:
            extracted[k] = pyd[k]
    tp_l1 = TaxPayer(**extracted)
    l1_answer, l1_trace = compute_answer(tp_l1)
    print("L1 (schema-only) answer: {}".format(l1_answer))
    print("Delta (L1 - GT): {}".format(l1_answer - gt_answer))

    # Drill down: toggle each missing field one at a time
    if missing_from_schema:
        print("\nField-by-field delta attribution:")
        for k in sorted(missing_from_schema):
            test_dict = dict(extracted)
            test_dict[k] = pyd[k]
            try:
                tp_test = TaxPayer(**test_dict)
                test_answer, _ = compute_answer(tp_test)
                field_delta = (l1_answer - test_answer)
                marker = " <--- NONZERO" if abs(field_delta) > 0.01 else ""
                print("    adding {:40s} changes answer by {:>10.2f}{}".format(
                    k, -field_delta, marker
                ))
            except Exception as e:
                print("    adding {:40s} ERROR: {}".format(k, e))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
