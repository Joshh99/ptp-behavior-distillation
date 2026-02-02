"""
Airline Baggage Fee Loader for RuleArena Benchmark

This loader provides data access and evaluation utilities for the airline domain
of the RuleArena benchmark. Ground truth computation uses the benchmark's 
reference implementation to ensure reproducibility and fair comparison.

Reference Implementation:
    Wang et al. "RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs 
    in Real-World Scenarios" ACL 2025
    GitHub: https://github.com/SkyRiver-2000/RuleArena
    
Citation:
    @inproceedings{zhou2025rulearena,
      title={RULEARENA: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios},
      author={Zhou, Ruiwen and Hua, Wenyue and Pan, Liangming and Cheng, Sitao and Wu, Xiaobao and Yu, En and Wang, William Yang},
      booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages={550--572},
      year={2025}
    }

Usage:
    loader = AirlineLoader("external/RuleArena")
    
    # Load problems
    problems = loader.load_problems(complexity_level=0, max_problems=10)
    
    # Compute ground truth (using RuleArena's reference implementation)
    for problem in problems:
        print(f"Problem: {problem.query}")
        print(f"Ground truth: ${problem.ground_truth}")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AirlineProblem:
    """
    Represents a single airline baggage fee problem from RuleArena.
    
    Attributes:
        id: Problem identifier (0-indexed within complexity level)
        query: Natural language query describing the passenger scenario
        ground_truth: Correct total cost in dollars (computed via reference implementation)
        info: Structured problem data (base_price, customer_class, routine, direction, bag_list)
        complexity_level: 0 (5 bags), 1 (8 bags), or 2 (11 bags)
    """
    id: int
    query: str
    ground_truth: int
    info: Dict[str, Any]
    complexity_level: int
    
    @property
    def num_bags(self) -> int:
        """Number of bags in this problem (excluding carry-on)."""
        return len(self.info['bag_list']) - 1  # First bag is carry-on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "info": self.info,
            "complexity_level": self.complexity_level,
        }


class AirlineLoader:
    """
    Loader for RuleArena Airline Baggage Fee benchmark.
    
    This loader:
    1. Loads synthesized problems from JSONL files (comp_0/1/2.jsonl)
    2. Computes ground truth using RuleArena's reference implementation
    3. Provides access to reference rules and fee tables
    
    The ground truth computation is delegated to RuleArena's `compute_answer.py`
    to ensure reproducibility and fair comparison with the benchmark paper.
    
    Attributes:
        repo_path: Path to cloned RuleArena repository
        airline_path: Path to airline subdirectory (repo_path/airline)
        
    Example:
        >>> loader = AirlineLoader("external/RuleArena")
        >>> problems = loader.load_problems(complexity_level=0, max_problems=5)
        >>> print(f"Loaded {len(problems)} problems")
        >>> print(f"First problem answer: ${problems[0].ground_truth}")
    """
    
    # Problem complexity mapping
    COMPLEXITY_FILES = {
        0: "comp_0.jsonl",  # 5 bags per problem
        1: "comp_1.jsonl",  # 8 bags per problem
        2: "comp_2.jsonl",  # 11 bags per problem
    }
    
    def __init__(self, repo_path: str):
        """
        Initialize the AirlineLoader.
        
        Args:
            repo_path: Path to cloned RuleArena repository.
                      Can be relative (e.g., "external/RuleArena") or absolute.
                      
        Raises:
            FileNotFoundError: If repository or airline subdirectory not found.
            ImportError: If RuleArena's compute_answer module cannot be imported.
        """
        self.repo_path = Path(repo_path)
        self.airline_path = self.repo_path / "airline"
        
        # Validate paths
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"RuleArena repository not found at: {self.repo_path}\n"
                f"Clone it with: git clone https://github.com/SkyRiver-2000/RuleArena {repo_path}"
            )
        
        if not self.airline_path.exists():
            raise FileNotFoundError(
                f"Airline subdirectory not found at: {self.airline_path}\n"
                f"Make sure the RuleArena repository is properly cloned."
            )
        
        # Import RuleArena's reference implementation
        self._import_reference_implementation()
    
    def _import_reference_implementation(self):
        """
        Import RuleArena's ground truth computation code.
        
        We use a vendored copy of their code (src/dataset/rulearena_reference.py)
        with clear attribution. This avoids Python import issues and is standard
        practice for research benchmarks.
        
        Raises:
            ImportError: If rulearena_reference module cannot be imported.
        """
        try:
            # Import vendored reference implementation
            # This should be in src/dataset/rulearena_reference.py
            from . import rulearena_reference
            
            self._compute_answer_fn = rulearena_reference.compute_answer
            self._fee_tables = rulearena_reference.load_checking_fee(str(self.repo_path))
            
            print(f"✓ Loaded RuleArena reference implementation (vendored copy)")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import rulearena_reference module.\n"
                f"Make sure rulearena_reference.py exists in src/dataset/\n"
                f"Error: {e}"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Failed to load fee tables from {self.airline_path}\n"
                f"Make sure the RuleArena repository has fee_tables/ directory.\n"
                f"Error: {e}"
            )
    
    def load_problems(
        self, 
        complexity_level: int, 
        max_problems: Optional[int] = None
    ) -> List[AirlineProblem]:
        """
        Load airline baggage fee problems from RuleArena dataset.
        
        Args:
            complexity_level: Difficulty level (0=easy, 1=medium, 2=hard)
                             - Level 0: 5 bags per problem (100 problems)
                             - Level 1: 8 bags per problem (100 problems)
                             - Level 2: 11 bags per problem (100 problems)
            max_problems: Optional limit on number of problems to load.
                         If None, loads all problems from the file.
        
        Returns:
            List of AirlineProblem objects with computed ground truth.
            
        Raises:
            ValueError: If complexity_level is not in {0, 1, 2}
            FileNotFoundError: If problem file doesn't exist
        """
        if complexity_level not in self.COMPLEXITY_FILES:
            raise ValueError(
                f"Invalid complexity_level: {complexity_level}. "
                f"Must be one of: {list(self.COMPLEXITY_FILES.keys())}"
            )
        
        filename = self.COMPLEXITY_FILES[complexity_level]
        filepath = self.airline_path / "synthesized_problems" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Problem file not found: {filepath}\n"
                f"Make sure the RuleArena repository is properly cloned."
            )
        
        problems = []
        
        print(f"Loading problems from {filename}...")
        
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
        
        print(f"✓ Loaded {len(problems)} problems (complexity level {complexity_level})")
        return problems
    
    def _parse_problem(
        self, 
        raw_problem: Dict[str, Any], 
        idx: int, 
        complexity_level: int
    ) -> AirlineProblem:
        """
        Parse raw problem and compute ground truth.
        
        Args:
            raw_problem: Dictionary with 'prompt' and 'info' keys from JSONL
            idx: Problem index (0-indexed)
            complexity_level: The complexity level (0/1/2)
        
        Returns:
            AirlineProblem with ground truth computed via reference implementation.
        """
        query = raw_problem['prompt']
        info = raw_problem['info']
        
        # Compute ground truth using RuleArena's reference implementation
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
        """
        Load airline baggage fee reference rules.
        
        Args:
            textual: If True, load textual IF-THEN format (for LLM parsing).
                    If False, load markdown table format (human-readable).
        
        Returns:
            String containing the reference rules.
            
        Raises:
            FileNotFoundError: If rules file doesn't exist.
        """
        filename = "reference_rules_textual.txt" if textual else "reference_rules.txt"
        filepath = self.airline_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Rules file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            rules = f.read()
        
        return rules
    
    def get_problem_count(self, complexity_level: int) -> int:
        """
        Get total number of problems for a complexity level.
        
        Args:
            complexity_level: 0, 1, or 2
            
        Returns:
            Number of problems in the file (typically 100 per level).
        """
        if complexity_level not in self.COMPLEXITY_FILES:
            raise ValueError(f"Invalid complexity_level: {complexity_level}")
        
        filename = self.COMPLEXITY_FILES[complexity_level]
        filepath = self.airline_path / "synthesized_problems" / filename
        
        if not filepath.exists():
            return 0
        
        with open(filepath, "r") as f:
            return sum(1 for _ in f)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_airline_problems(
    repo_path: str = "external/RuleArena",
    complexity_level: int = 0,
    max_problems: Optional[int] = None,
) -> List[AirlineProblem]:
    """
    Convenience function to quickly load airline problems.
    
    Args:
        repo_path: Path to RuleArena repository
        complexity_level: 0 (easy), 1 (medium), or 2 (hard)
        max_problems: Optional limit on problems to load
        
    Returns:
        List of AirlineProblem objects
        
    Example:
        >>> problems = load_airline_problems(complexity_level=0, max_problems=5)
        >>> print(f"First problem: ${problems[0].ground_truth}")
    """
    loader = AirlineLoader(repo_path)
    return loader.load_problems(complexity_level, max_problems)


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    # When run as a script, we need to import the vendored module differently
    import sys
    from pathlib import Path
    
    # Add parent directory to path so we can import rulearena_reference
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("=" * 80)
    print("AirlineLoader Test")
    print("=" * 80)
    
    try:
        # Initialize loader
        loader = AirlineLoader("external/RuleArena")
        print()
        
        # Load a few problems
        print("Loading 3 problems from complexity level 0...")
        problems = loader.load_problems(complexity_level=0, max_problems=3)
        print()
        
        # Display first problem
        if problems:
            print("First Problem:")
            print("-" * 80)
            p = problems[0]
            print(f"ID: {p.id}")
            print(f"Query: {p.query[:150]}...")
            print(f"Ground Truth: ${p.ground_truth}")
            print(f"Complexity: Level {p.complexity_level}")
            print(f"Number of bags: {p.num_bags}")
            print(f"Customer class: {p.info['customer_class']}")
            print(f"Route: {p.info['routine']}")
            print(f"Direction: {'Arriving to US' if p.info['direction'] else 'Departing from US'}")
            print()
        
        # Load rules
        print("Loading reference rules...")
        rules = loader.load_rules(textual=False)
        print(f"✓ Rules loaded: {len(rules)} characters")
        print(f"Preview: {rules[:200]}...")
        print()
        
        # Problem counts
        print("Problem counts by complexity:")
        for level in [0, 1, 2]:
            count = loader.get_problem_count(level)
            print(f"  Level {level}: {count} problems")
        
        print()
        print("=" * 80)
        print("All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()