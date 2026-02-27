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

            # Build messages — airline gets a domain-specific system prompt
            messages = []
            if instance.domain == "airline":
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a helpful assistant at American Airlines. "
                        "You are given the information of a passenger, his / her items, "
                        "his / her special needs, and the policies of American Airlines. "
                        "You should compute the total cost (including the flight ticket fee, "
                        "checked bag fees, cost of special needs) according to the policies "
                        "for the passenger."
                    ),
                })
            messages.append({"role": "user", "content": prompt})

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )

            # Extract response
            raw_response = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Parse answer
            predicted = self._parse_answer(raw_response, instance.domain)

            # Compare with ground truth
            exact_match, tolerance_match = self.compare_answers(
                predicted, instance.ground_truth_answer
            )

            # Calculate cost
            cost = calculate_cost(input_tokens, output_tokens)

            # Calculate time
            elapsed = time.time() - start_time

            # Determine failure mode
            if predicted is None:
                failure_mode = "extraction_failure"
            elif exact_match:
                failure_mode = "none"
            else:
                failure_mode = "calculation_error"

            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=predicted,
                expected=instance.ground_truth_answer,
                is_correct_exact=exact_match,
                is_correct_tolerance=tolerance_match,
                latency_ms=elapsed * 1000,
                cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                calculator_name=instance.domain,
                error=None,
                failure_mode=failure_mode,
                raw_response=raw_response,
                metadata={
                    "domain": instance.domain,
                    "complexity_level": instance.complexity_level,
                }
            )

        except Exception as e:
            # Robust error handling - never crash
            elapsed = time.time() - start_time
            return ExperimentResult(
                instance_id=instance.instance_id,
                predicted=None,
                expected=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                latency_ms=elapsed * 1000,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                calculator_name=instance.domain,
                error=str(e),
                failure_mode="extraction_failure",
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

        if instance.domain == "nba":
            return f"""You are a helpful NBA team consultant.

{instance.rules_text}

{instance.problem_text}

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter.
Your response:
"""

        # Airline: match RuleArena reference prompt structure exactly
        _AIRLINE_EXAMPLE = """\
Here is an example for you to follow:
<example>
John is a Main Plus Class passenger flying from Orlando to Lima with the following items:
1. A backpack: 21 x 11 x 7 inches, 9 lbs;
2. A luggage box: 51 x 24 x 27 inches, 65 lbs;
3. A backpack: 46 x 29 x 24 inches, 85 lbs;
4. A luggage box: 35 x 12 x 11 inches, 44 lbs;
5. A luggage box: 30 x 11 x 10 inches, 33 lbs;
John's flight ticket is $448. Compute the total cost for him step by step and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
To calculate the total cost for John, we need to consider the flight ticket cost, checked bag fees, and any overweight or oversize fees according to American Airlines' policies.

Flight Ticket Cost:
- John's flight ticket: $448

Carry-on Allowance:
- John is allowed 1 carry-on bag and 1 personal item.
- Item 1 (Backpack):
  - Dimensions: 21 x 11 x 7 inches
  - Weight: 9 lbs
  - This fits within the carry-on size limit of 22 x 14 x 9 inches.
  - Fee: $0 (Allowed as carry-on bag)
- John does not have any items that fit the personal item dimensions (18 x 14 x 8 inches). So, no personal item is carried.

Checked Bags:
- Items to be checked: Items 2, 3, 4, and 5
- John is a Main Plus passenger, which includes 1 extra free checked bag in addition to the Main Cabin allowance, for a total of 2 free checked bags.
- Checked Bag Fees:
  - First Bag: $0 (free)
  - Second Bag: $0 (free)
  - Third Bag: $200
  - Fourth Bag: $200

Fees for Each Checked Bag:

1. Item 2 (Luggage box):
   - Dimensions: 51 x 24 x 27 inches
     - Total dimensions: 51 + 24 + 27 = 102 inches
     - Over the standard size limit of 62 inches.
   - Weight: 65 lbs
     - Over the standard weight limit of 50 lbs but under 70 lbs.
   - Checking Fee:
     - For the first checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 53 lbs up to 70 lbs, the fee is $100.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 2: $0 (checking) + $150 (oversize) = $150
2. Item 3 (Backpack):
   - Dimensions: 46 x 29 x 24 inches
     - Total dimensions: 46 + 29 + 24 = 99 inches
     - Over the standard size limit of 62 inches.
   - Weight: 85 lbs
     - Over the standard weight limit of 50 lbs and over 70 lbs but under 100 lbs.
   - Checking Fee:
     - For the second checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 70 lbs up to 100 lbs, the fee is $200.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 3: $0 (checking) + $200 (overweight) = $200
3. Item 4 (Luggage box):
   - Dimensions: 35 x 12 x 11 inches
     - Total dimensions: 35 + 12 + 11 = 58 inches
     - Within the standard size limit of 62 inches.
   - Weight: 44 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the third checked bag, the checking fee is $200.
   - Total Fee for Item 4: $200 (checking) + $0 (No overweight or oversize fees) = $200
4. Item 5 (Luggage box):
   - Dimensions: 30 x 11 x 10 inches
     - Total dimensions: 30 + 11 + 10 = 51 inches
     - Within the standard size limit of 62 inches.
   - Weight: 33 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the fourth checked bag, the checking fee is $200.
   - Total Fee for Item 5: $200 (checking) + $0 (No overweight or oversize fees) = $200
Summary of Baggage Fees:
  - Item 2: $150
  - Item 3: $200
  - Item 4: $200
  - Item 5: $200
Total Baggage Fees: $150 (Item 2) + $200 (Item 3) + $200 (Item 4) + $200 (Item 5) = $750
Total Cost:
- Flight Ticket: $448
- Total Baggage Fees: $750
The total cost is $1,198.
</example>
"""
        return (
            f"You should compute the total cost (including the flight ticket fee, "
            f"checked bag fees, cost of special needs) according to the policies.\n\n"
            f"The policies of American Airlines are as follows:\n\n"
            f"{instance.rules_text}\n"
            f"{_AIRLINE_EXAMPLE}\n"
            f"{instance.problem_text} Compute the total cost step by step "
            f"(don't omit any bag) and end your response with "
            f'"The total cost is $xxx." (xxx is a number)\n'
            f"Your response:\n"
        )

    def _parse_answer(self, response: str, domain: str = None):
        """
        Parse answer from LLM response.

        For NBA: returns True or False (boolean).
        For tax/airline: returns float or None.
        """
        if not response:
            return None

        # Strip markdown bold markers before any regex (matches auto_test.py:168)
        response = response.replace("**", "")

        # NBA: boolean classification — check for exact Answer: True/False strings
        if domain == "nba":
            if "Answer: True" in response:
                return True
            if "Answer: False" in response:
                return False
            return None

        # Airline: "The total cost is $xxx." (matches reference auto_test.py extraction)
        if domain == "airline":
            airline_match = re.search(
                r'The total cost is \$([\d,]+(?:\.\d+)?)', response
            )
            if airline_match:
                try:
                    return float(airline_match.group(1).replace(',', ''))
                except ValueError:
                    pass

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
            r'The total tax (owed|overpaid) is \$([\d,]+(?:\.\d+)?)',
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
    print(f"  Correct: {result.is_correct_exact}")
    print(f"  Cost: ${result.cost_usd:.4f}")
    print(f"  Time: {result.latency_ms:.0f}ms")
    if result.error:
        print(f"  Error: {result.error}")
