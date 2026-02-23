"""
L0F: Chain-of-Thought Baseline Experiment

Pure LLM reasoning with no structure. Single API call with CoT prompting.
Replicates the original RuleArena paper's evaluation methodology.
"""

import time
import re
from pathlib import Path
from typing import Dict, Optional

from together import Together

from benchmark.rulearena.dataset.loader import RuleArenaInstance, RULEARENA_PATH
from benchmark.rulearena.experiments.base import BaseExperiment, ExperimentResult
from benchmark.rulearena.config import MODEL_CONFIG, calculate_cost

# ---------------------------------------------------------------------------
# Tax form templates (loaded once at import time)
# ---------------------------------------------------------------------------
_tax_prompt_module = None

def _get_tax_templates():
    """Lazy-load tax form templates from external/RuleArena/tax/prompt.py."""
    global _tax_prompt_module
    if _tax_prompt_module is not None:
        return _tax_prompt_module

    import importlib.util
    prompt_path = RULEARENA_PATH / "tax" / "prompt.py"
    spec = importlib.util.spec_from_file_location("tax_prompt", prompt_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _tax_prompt_module = mod
    return mod


# Matches auto_test.py:30-38
TAX_PROMPT_TEMPLATE = """
You are given several forms used to report US income tax and the instructions or rules about how to fill the forms. Then you will be given the income and/or payment information about a tax payer According to the given information. You should calculate the income tax owed by this payer.

IRS Forms for the tax payer:
$forms
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
"""


def build_tax_query(problem_data: Dict) -> str:
    """Build the full tax query by filling IRS form templates with taxpayer data.

    Replicates the logic from external/RuleArena/tax/auto_test.py:118-152.

    Args:
        problem_data: Full problem dict with 'dict' and 'pydantic' keys.

    Returns:
        Complete prompt string with filled IRS forms.
    """
    mod = _get_tax_templates()
    tax_payer = problem_data["dict"]

    # 1. Select relevant form templates based on taxpayer flags
    forms = [mod.basic_forms]
    if tax_payer["itemized"]:
        forms.append(mod.itemized_forms)
    if tax_payer["self_employed"]:
        forms.append(mod.self_employ_forms)
    if tax_payer["has_student_loans_or_education_expenses"]:
        forms.append(mod.edu_forms)
    if tax_payer["child_and_dependent"]:
        forms.append(mod.schedule_8812)
    forms = "".join(forms)

    # 2. Substitute data values (numeric -> "$1,234", string -> as-is)
    tbd_fields = []
    for k, v in tax_payer["data"].items():
        if isinstance(v, str):
            forms = forms.replace("$" + k, v)
        else:
            forms = forms.replace("$" + k, "$" + f"{v:,}")
        if v == "$TBD":
            tbd_fields.append(k)

    # 3. Replace remaining $TBD with [__]
    forms = forms.replace("$TBD", "[__]")

    # 4. Fill demographic fields
    prompt = TAX_PROMPT_TEMPLATE.replace("$forms", forms)
    prompt = prompt.replace("$name", tax_payer["name"])
    prompt = prompt.replace("$age", str(tax_payer["age"]))
    prompt = prompt.replace("$spouse_age", str(tax_payer["spouse_age"]))
    prompt = prompt.replace("$blind", str(tax_payer["blind"]))
    prompt = prompt.replace("$spouse_blind", str(tax_payer["spouse_blind"]))
    prompt = prompt.replace("$filing_status", tax_payer["filing_status"])
    prompt = prompt.replace("$itemized", str(tax_payer["itemized"]))
    prompt = prompt.replace("$num_qualifying_children", str(tax_payer["num_qualifying_children"]))
    prompt = prompt.replace("$num_other_dependents", str(tax_payer["num_other_dependents"]))

    return prompt


class L0F_CoT_Experiment(BaseExperiment):
    """
    L0F: Chain-of-Thought baseline.

    Single LLM call with rules + problem + CoT instruction.
    No structure, no tools - just pure reasoning.
    """

    def __init__(self):
        super().__init__(
            experiment_name="l0f_cot",
            model_id=MODEL_CONFIG["model_id"]
        )
        self.client = Together(api_key=None)  # Uses env var TOGETHER_API_KEY
        self.temperature = MODEL_CONFIG["default_params"]["temperature"]
        self.seed = MODEL_CONFIG["default_params"]["seed"]
        self.max_tokens = MODEL_CONFIG["default_params"]["max_tokens"]

    def run_instance(self, instance: RuleArenaInstance) -> ExperimentResult:
        """
        Run L0F experiment on a single instance.

        Returns:
            ExperimentResult with prediction and metrics
        """
        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_prompt(instance)

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )

            # Extract response
            raw_response = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Parse answer
            predicted = self._parse_answer(raw_response)

            # Compare with ground truth
            exact_match, tolerance_match = self.compare_answers(
                predicted, instance.ground_truth_answer
            )

            # Calculate cost
            cost = calculate_cost(input_tokens, output_tokens)

            # Calculate time
            elapsed = time.time() - start_time

            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=predicted,
                expected=instance.ground_truth_answer,
                correct=exact_match,
                time_seconds=elapsed,
                cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=None,
                raw_response=raw_response,
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                    "tolerance_match": tolerance_match,
                }
            )

        except Exception as e:
            # Robust error handling - never crash
            elapsed = time.time() - start_time
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                correct=False,
                time_seconds=elapsed,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                error=str(e),
                raw_response=None,
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                }
            )

    def _build_prompt(self, instance: RuleArenaInstance) -> str:
        """
        Build CoT prompt for the instance.

        Format matches RuleArena paper's methodology.
        For tax domain, constructs the query from IRS form templates.
        """
        if instance.domain == "tax":
            return build_tax_query(instance.metadata)

        prompt = f"""You are solving a {instance.domain} calculation problem.

RULES:
{instance.rules_text[:2000]}

PROBLEM:
{instance.problem_text}

Think step by step. Show your reasoning, then give your final numeric answer on the last line as:
ANSWER: <number>
"""
        return prompt

    def _parse_answer(self, response: str) -> Optional[float]:
        """
        Parse answer from LLM response.

        Looks for "ANSWER: <number>" pattern, tax-specific patterns, or
        extracts last number as fallback.
        """
        if not response:
            return None

        # Try to find "ANSWER: <number>" pattern
        answer_match = re.search(r'ANSWER:\s*\$?\s*([\d,]+\.?\d*)', response, re.IGNORECASE)
        if answer_match:
            try:
                answer_str = answer_match.group(1).replace(',', '')
                return float(answer_str)
            except ValueError:
                pass

        # Tax-specific: "The total tax owed/overpaid is $xxx."
        tax_match = re.search(
            r'The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?)',
            response
        )
        if tax_match:
            try:
                value = float(tax_match.group(2).replace(',', ''))
                if tax_match.group(1) == "overpaid":
                    value = -value
                return value
            except ValueError:
                pass

        # Fallback: extract last number in response
        numbers = re.findall(r'\$?\s*([\d,]+\.?\d+)', response)
        if numbers:
            try:
                # Take the last number (often the final answer)
                answer_str = numbers[-1].replace(',', '')
                return float(answer_str)
            except ValueError:
                pass

        return None


if __name__ == "__main__":
    # Quick test
    from benchmark.rulearena.dataset.loader import RuleArenaDataset

    print("Testing L0F CoT Experiment...")
    dataset = RuleArenaDataset()
    sample = dataset.stratified_sample(1, seed=42)

    experiment = L0F_CoT_Experiment()
    result = experiment.run_instance(sample[0])

    print(f"\nResult:")
    print(f"  Instance: {result.instance_id}")
    print(f"  Predicted: {result.predicted}")
    print(f"  Expected: {result.expected}")
    print(f"  Correct: {result.correct}")
    print(f"  Cost: ${result.cost_usd:.4f}")
    print(f"  Time: {result.time_seconds:.2f}s")
    if result.error:
        print(f"  Error: {result.error}")
