"""
RuleArena Dataset Loader

Loads problems from all three RuleArena domains (airline, nba, tax) and provides
stratified sampling across domains and complexity levels.

Matches MedCalc's dataclass structure for cross-benchmark compatibility.
"""

import json
import sys
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Add RuleArena to path for reference implementations
REPO_ROOT = Path(__file__).parent.parent.parent.parent
RULEARENA_PATH = REPO_ROOT / "external" / "RuleArena"

# Import domain calculators
try:
    from benchmark.rulearena.calculators.airline import compute_airline_fee
    AIRLINE_COMPUTE_FN = compute_airline_fee
except Exception as e:
    print(f"Warning: Could not load airline calculator: {e}")
    AIRLINE_COMPUTE_FN = None

try:
    from benchmark.rulearena.calculators.tax import compute_tax_fee
    TAX_COMPUTE_FN = compute_tax_fee
except Exception as e:
    print(f"Warning: Could not load tax calculator: {e}")
    TAX_COMPUTE_FN = None


@dataclass
class RuleArenaInstance:
    """
    Single benchmark instance. Mirrors MedCalcInstance structure.

    Schema MUST match MedCalc's schema for cross-benchmark analysis.
    """
    instance_id: str                # Unique identifier: "{domain}_{complexity}_{idx}"
    domain: str                     # "airline" | "nba" | "tax"
    complexity_level: int           # 0, 1, 2
    problem_text: str               # The natural language question
    rules_text: str                 # Domain rules provided to the model
    ground_truth_answer: Any        # Expected numeric/string answer
    ground_truth_explanation: str   # Step-by-step solution (if available)
    metadata: Dict[str, Any]        # Additional fields from RuleArena JSON

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "domain": self.domain,
            "complexity_level": self.complexity_level,
            "problem_text": self.problem_text,
            "rules_text": self.rules_text,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_explanation": self.ground_truth_explanation,
            "metadata": self.metadata,
        }


class RuleArenaDataset:
    """
    Dataset loader for RuleArena benchmark.

    Loads all 816 problems across 3 domains and 3 complexity levels.
    Provides stratified sampling for balanced evaluation.
    """

    DOMAINS = ["airline", "nba", "tax"]
    COMPLEXITY_LEVELS = [0, 1, 2]

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize dataset loader.

        Args:
            repo_path: Path to RuleArena repo (default: external/RuleArena)
        """
        if repo_path is None:
            repo_path = RULEARENA_PATH

        self.repo_path = Path(repo_path)
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"RuleArena repository not found at {self.repo_path}. "
                f"Clone from: https://github.com/SkyRiver-2000/RuleArena"
            )

        self.instances: List[RuleArenaInstance] = []
        self._load_all_instances()

    def _load_all_instances(self):
        """Load all instances from all domains and complexity levels."""
        for domain in self.DOMAINS:
            for complexity in self.COMPLEXITY_LEVELS:
                instances = self._load_domain_complexity(domain, complexity)
                self.instances.extend(instances)

        print(f"Loaded {len(self.instances)} instances from RuleArena")

    def _load_domain_complexity(
        self, domain: str, complexity: int
    ) -> List[RuleArenaInstance]:
        """Load instances for a specific domain and complexity level."""
        instances = []

        # Determine path based on domain
        if domain == "airline":
            data_dir = self.repo_path / "airline" / "synthesized_problems"
            rules_file = self.repo_path / "airline" / "reference_rules_textual.txt"
            file_ext = "jsonl"
        elif domain == "nba":
            data_dir = self.repo_path / "nba" / "annotated_problems"
            rules_file = self.repo_path / "nba" / "reference_rules.txt"
            file_ext = "json"  # NBA uses .json not .jsonl
        elif domain == "tax":
            data_dir = self.repo_path / "tax" / "synthesized_problems"
            rules_file = self.repo_path / "tax" / "prompt.py"  # Tax doesn't have txt rules
            file_ext = "json"  # Tax uses .json not .jsonl
        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Load rules if available
        if rules_file.exists():
            rules_text = rules_file.read_text(encoding='utf-8')
        else:
            rules_text = f"Rules for {domain} domain (see original RuleArena repo)"

        # Load problems
        problem_file = data_dir / f"comp_{complexity}.{file_ext}"
        if not problem_file.exists():
            print(f"Warning: {problem_file} not found, skipping")
            return instances

        # Load based on file format
        if file_ext == "jsonl":
            with open(problem_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    problem_data = json.loads(line)
                    instance = self._create_instance(
                        domain, complexity, idx, problem_data, rules_text
                    )
                    instances.append(instance)
        else:  # json
            with open(problem_file, 'r', encoding='utf-8') as f:
                problems_list = json.load(f)
                for idx, problem_data in enumerate(problems_list):
                    instance = self._create_instance(
                        domain, complexity, idx, problem_data, rules_text
                    )
                    instances.append(instance)

        return instances

    def _create_instance(
        self, domain: str, complexity: int, idx: int,
        problem_data: Dict, rules_text: str
    ) -> RuleArenaInstance:
        """Create a RuleArenaInstance from problem data."""
        # Tax and NBA problems have no 'info' key — store full problem_data
        # so experiments can access 'dict'/'pydantic' (tax) or
        # 'team_situations'/'player_situations'/'operations' (NBA).
        if domain in ("tax", "nba"):
            metadata = problem_data
        else:
            metadata = problem_data.get('info', {})

        # Compute ground truth if possible
        ground_truth = self._compute_ground_truth(domain, problem_data, metadata)

        return RuleArenaInstance(
            instance_id=f"{domain}_{complexity}_{idx}",
            domain=domain,
            complexity_level=complexity,
            problem_text=problem_data.get('prompt', ''),
            rules_text=rules_text,
            ground_truth_answer=ground_truth,
            ground_truth_explanation=problem_data.get('explanation', ''),
            metadata=metadata,
        )

    def _compute_ground_truth(self, domain: str, problem_data: Dict, metadata: Dict) -> Any:
        """
        Compute ground truth using domain-specific calculators.

        Args:
            domain: Domain name (airline, nba, tax)
            problem_data: Raw problem dict from the JSON file
            metadata: Processed metadata stored on the instance

        Returns:
            Ground truth answer or None if computation fails
        """
        try:
            if domain == "airline" and AIRLINE_COMPUTE_FN is not None:
                return AIRLINE_COMPUTE_FN(metadata)
            elif domain == "tax" and TAX_COMPUTE_FN is not None:
                return TAX_COMPUTE_FN(metadata)
            elif domain == "nba":
                return problem_data.get('answer')
            else:
                return None
        except Exception as e:
            return None


    def stratified_sample(
        self, n: int, seed: int = 42
    ) -> List[RuleArenaInstance]:
        """
        Sample n instances with stratification across domains and complexity.

        Args:
            n: Number of instances to sample
            seed: Random seed for reproducibility

        Returns:
            List of n instances, stratified by domain and complexity
        """
        random.seed(seed)

        # Group instances by (domain, complexity)
        groups: Dict[tuple, List[RuleArenaInstance]] = {}
        for inst in self.instances:
            key = (inst.domain, inst.complexity_level)
            if key not in groups:
                groups[key] = []
            groups[key].append(inst)

        # Calculate samples per group
        num_groups = len(groups)
        samples_per_group = n // num_groups
        remainder = n % num_groups

        # Sample from each group
        sampled = []
        for idx, (key, group_instances) in enumerate(sorted(groups.items())):
            # Distribute remainder across first groups
            group_n = samples_per_group + (1 if idx < remainder else 0)
            group_sample = random.sample(
                group_instances, min(group_n, len(group_instances))
            )
            sampled.extend(group_sample)

        # Shuffle final sample
        random.shuffle(sampled)
        return sampled[:n]

    def get_by_domain(self, domain: str) -> List[RuleArenaInstance]:
        """Get all instances for a specific domain."""
        return [inst for inst in self.instances if inst.domain == domain]

    def get_by_complexity(self, complexity: int) -> List[RuleArenaInstance]:
        """Get all instances for a specific complexity level."""
        return [
            inst for inst in self.instances if inst.complexity_level == complexity
        ]

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> RuleArenaInstance:
        return self.instances[idx]


if __name__ == "__main__":
    # Test the loader
    dataset = RuleArenaDataset()
    print(f"Total instances: {len(dataset)}")
    print(f"Domains: {set(inst.domain for inst in dataset.instances)}")
    print(f"Complexity levels: {set(inst.complexity_level for inst in dataset.instances)}")

    # Test stratified sampling
    sample = dataset.stratified_sample(9, seed=42)
    print(f"\nStratified sample (9 instances):")
    for inst in sample:
        print(f"  {inst.instance_id} | {inst.domain} | level={inst.complexity_level}")
