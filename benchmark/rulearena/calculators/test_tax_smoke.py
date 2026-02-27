import sys
from pathlib import Path

# Add project root to sys.path so "benchmark.*" imports resolve
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from benchmark.rulearena.calculators.tax import compute_tax_fee
from structured_forms import TaxPayer
from micro_evaluation import compute_answer

with open(PROJECT_ROOT / 'external/RuleArena/tax/synthesized_problems/comp_0.json') as f:
    problems = json.load(f)

for i, problem in enumerate(problems[:3]):
    result = compute_tax_fee(problem)
    tp = TaxPayer(**problem['pydantic'])
    expected, _ = compute_answer(tp)
    match = 'MATCH' if result == float(expected) else 'MISMATCH'
    print(f'Problem {i}: computed={result}, expected={expected}, {match}')
