import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from benchmark.rulearena import rulearena_reference


@dataclass
class AirlineProblem:
    id: int
    query: str
    ground_truth: int
    info: Dict[str, Any]
    complexity_level: int

    @property
    def num_bags(self) -> int:
        return len(self.info['bag_list']) - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "info": self.info,
            "complexity_level": self.complexity_level,
        }


class AirlineLoader:
    COMPLEXITY_FILES = {
        0: "comp_0.jsonl",
        1: "comp_1.jsonl",
        2: "comp_2.jsonl",
    }

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.airline_path = self.repo_path / "airline"

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"RuleArena repository not found at: {self.repo_path}\n"
                f"Clone it with: git clone https://github.com/SkyRiver-2000/RuleArena {repo_path}"
            )

        if not self.airline_path.exists():
            raise FileNotFoundError(
                f"Airline subdirectory not found at: {self.airline_path}"
            )

        self._compute_answer_fn = rulearena_reference.compute_answer
        self._fee_tables = rulearena_reference.load_checking_fee(str(self.repo_path))

    def load_problems(
        self,
        complexity_level: int,
        max_problems: Optional[int] = None
    ) -> List[AirlineProblem]:
        if complexity_level not in self.COMPLEXITY_FILES:
            raise ValueError(
                f"Invalid complexity_level: {complexity_level}. "
                f"Must be one of: {list(self.COMPLEXITY_FILES.keys())}"
            )

        filename = self.COMPLEXITY_FILES[complexity_level]
        filepath = self.airline_path / "synthesized_problems" / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Problem file not found: {filepath}")

        problems = []

        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_problems is not None and idx >= max_problems:
                    break
                try:
                    raw_problem = json.loads(line.strip())
                    problem = self._parse_problem(raw_problem, idx, complexity_level)
                    problems.append(problem)
                except Exception as e:
                    print(f"Warning: Failed to parse problem {idx}: {e}")
                    continue

        return problems

    def _parse_problem(
        self,
        raw_problem: Dict[str, Any],
        idx: int,
        complexity_level: int
    ) -> AirlineProblem:
        query = raw_problem['prompt']
        info = raw_problem['info']

        ground_truth, _ = self._compute_answer_fn(
            base_price=info['base_price'],
            direction=info['direction'],
            routine=info['routine'],
            customer_class=info['customer_class'],
            bag_list=info['bag_list'],
            check_base_tables=self._fee_tables,
        )

        return AirlineProblem(
            id=idx,
            query=query,
            ground_truth=ground_truth,
            info=info,
            complexity_level=complexity_level,
        )

    def load_rules(self, textual: bool = False) -> str:
        filename = "reference_rules_textual.txt" if textual else "reference_rules.txt"
        filepath = self.airline_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Rules file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def get_problem_count(self, complexity_level: int) -> int:
        if complexity_level not in self.COMPLEXITY_FILES:
            raise ValueError(f"Invalid complexity_level: {complexity_level}")

        filename = self.COMPLEXITY_FILES[complexity_level]
        filepath = self.airline_path / "synthesized_problems" / filename

        if not filepath.exists():
            return 0

        with open(filepath, "r") as f:
            return sum(1 for _ in f)
