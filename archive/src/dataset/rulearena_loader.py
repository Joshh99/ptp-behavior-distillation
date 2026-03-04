"""
RuleArena Data Loader

Loads airline baggage fee problems from the RuleArena benchmark dataset.
This loader works with the actual RuleArena repository structure and computes
ground truth answers using the benchmark's own calculation logic.

Repository: https://github.com/SkyRiver-2000/RuleArena

Problem Structure:
- comp_0.jsonl: 100 problems with 5 bags each (easiest)
- comp_1.jsonl: 100 problems with 8 bags each (medium)
- comp_2.jsonl: 100 problems with 11 bags each (hardest)

Usage:
    loader = RuleArenaLoader("external/RuleArena")
    problems = loader.load_airline_problems(complexity_level=0, max_problems=10)
    rules = loader.load_rules()
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AirlineProblem:
    """Represents a single airline baggage fee problem."""
    id: int
    query: str
    ground_truth: int
    passenger_context: Dict[str, Any]
    difficulty_level: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "passenger_context": self.passenger_context,
            "difficulty_level": self.difficulty_level,
        }


class RuleArenaLoader:
    """
    Loader for the RuleArena airline baggage fee benchmark dataset.
    
    This class provides methods to:
    - Load synthesized problems from JSONL files
    - Compute ground truth answers using the benchmark's fee calculation logic
    - Load reference rules in text format
    - Access fee lookup tables
    
    Attributes:
        repo_path: Path to the cloned RuleArena repository
        airline_path: Path to the airline subdirectory
        fee_tables: Loaded CSV fee tables (lazy-loaded on first use)
    """
    
    # Complexity level to filename mapping
    COMPLEXITY_FILES = {
        0: "comp_0.jsonl",
        1: "comp_1.jsonl",
        2: "comp_2.jsonl",
    }
    
    # Regions where First class gets complementary first bag
    COMPLEMENTARY_FIRST_REGIONS = [
        "China", "Hong Kong", "Japan", "South Korea", "India",
        "Qatar", "Haiti", "Cuba", "Panama", "Colombia",
        "Ecuador", "Peru", "South America", "Israel", "Europe",
        "Australia", "New Zealand"
    ]
    
    def __init__(self, repo_path: str):
        """
        Initialize the RuleArena loader.
        
        Args:
            repo_path: Path to the cloned RuleArena repository.
                       Can be relative (e.g., "external/RuleArena") or absolute.
        
        Raises:
            FileNotFoundError: If the repository path doesn't exist.
        """
        self.repo_path = Path(repo_path)
        self.airline_path = self.repo_path / "airline"
        
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
        
        # Lazy-loaded fee tables
        self._fee_tables: Optional[List[Dict[int, pd.DataFrame]]] = None
    
    @property
    def fee_tables(self) -> List[Dict[int, pd.DataFrame]]:
        """
        Load and cache the fee lookup tables.
        
        Returns:
            List of dictionaries, one per bag number (1-4).
            Each dictionary has keys 0 (departure) and 1 (arrival) 
            mapping to pandas DataFrames with fees by region and class.
        """
        if self._fee_tables is None:
            self._fee_tables = self._load_fee_tables()
        return self._fee_tables
    
    def _load_fee_tables(self) -> List[Dict[int, pd.DataFrame]]:
        """
        Load CSV fee tables from the fee_tables directory.
        
        The fee tables are organized as:
        - fee_tables/bag_{1,2,3,4}/ - one folder per bag number
        - 0.csv: fees when departing from U.S.
        - 1.csv: fees when arriving to U.S.
        
        Returns:
            List of 4 dictionaries (bag 1-4), each with direction keys (0, 1).
        """
        fee_tables = []
        fee_path = self.airline_path / "fee_tables"
        
        for bag_num in range(1, 5):
            bag_folder = fee_path / f"bag_{bag_num}"
            
            if not bag_folder.exists():
                print(f"Warning: Fee table folder not found: {bag_folder}")
                fee_tables.append({0: pd.DataFrame(), 1: pd.DataFrame()})
                continue
            
            direction_tables = {}
            for direction in [0, 1]:
                csv_path = bag_folder / f"{direction}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path, index_col=0)
                        direction_tables[direction] = df
                    except Exception as e:
                        print(f"Warning: Failed to load {csv_path}: {e}")
                        direction_tables[direction] = pd.DataFrame()
                else:
                    print(f"Warning: Fee table not found: {csv_path}")
                    direction_tables[direction] = pd.DataFrame()
            
            fee_tables.append(direction_tables)
        
        return fee_tables
    
    def load_airline_problems(
        self, 
        complexity_level: int, 
        max_problems: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load airline baggage fee problems for a given complexity level.
        
        Args:
            complexity_level: Problem complexity (0=easy/5 bags, 1=medium/8 bags, 2=hard/11 bags)
            max_problems: Maximum number of problems to load (None = all)
        
        Returns:
            List of dictionaries with keys:
            - id: Problem identifier
            - query: Natural language problem description
            - ground_truth: Computed total cost (ticket + fees)
            - passenger_context: Dictionary with passenger/bag details
            - difficulty_level: The complexity level (0, 1, or 2)
        
        Raises:
            ValueError: If complexity_level is not 0, 1, or 2.
            FileNotFoundError: If the JSONL file doesn't exist.
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
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if max_problems is not None and idx >= max_problems:
                        break
                    
                    try:
                        raw_problem = json.loads(line.strip())
                        problem = self._parse_problem(raw_problem, idx, complexity_level)
                        problems.append(problem)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {idx} in {filename}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to process problem {idx} in {filename}: {e}")
                        continue
        except Exception as e:
            raise RuntimeError(f"Failed to read problem file {filepath}: {e}")
        
        return problems
    
    def _parse_problem(
        self, 
        raw_problem: Dict[str, Any], 
        idx: int, 
        complexity_level: int
    ) -> Dict[str, Any]:
        """
        Parse a raw problem from JSONL and compute ground truth.
        
        Args:
            raw_problem: Dictionary with 'prompt' and 'info' keys
            idx: Problem index
            complexity_level: The complexity level of this problem
        
        Returns:
            Parsed problem dictionary with ground truth computed.
        """
        prompt = raw_problem.get("prompt", "")
        info = raw_problem.get("info", {})
        
        # Extract passenger context
        passenger_context = {
            "base_price": info.get("base_price", 0),
            "customer_class": info.get("customer_class", "Main Cabin"),
            "routine": info.get("routine", "U.S."),  # Region/destination
            "direction": info.get("direction", 0),    # 0=outbound, 1=return
            "bag_list": info.get("bag_list", []),
            "num_bags": len(info.get("bag_list", [])),
        }
        
        # Compute ground truth
        try:
            ground_truth = self._compute_answer(
                base_price=passenger_context["base_price"],
                direction=passenger_context["direction"],
                routine=passenger_context["routine"],
                customer_class=passenger_context["customer_class"],
                bag_list=passenger_context["bag_list"],
            )
        except Exception as e:
            print(f"Warning: Failed to compute ground truth for problem {idx}: {e}")
            ground_truth = -1  # Indicate computation failure
        
        return {
            "id": idx,
            "query": prompt,
            "ground_truth": ground_truth,
            "passenger_context": passenger_context,
            "difficulty_level": complexity_level,
        }
    
    def _compute_answer(
        self,
        base_price: int,
        direction: int,
        routine: str,
        customer_class: str,
        bag_list: List[Dict[str, Any]],
    ) -> int:
        """
        Compute the total cost (ticket + baggage fees) for a passenger.
        
        This implements the same logic as RuleArena's compute_answer.py.
        
        Args:
            base_price: Ticket price in USD
            direction: 0 = departing from U.S., 1 = arriving to U.S.
            routine: Region/destination (e.g., "U.S.", "Europe", "China")
            customer_class: Cabin class (e.g., "Main Cabin", "Business")
            bag_list: List of bags with 'size' [L,W,H] and 'weight' keys
        
        Returns:
            Total cost in USD (ticket + all baggage fees).
        """
        # Skip the first bag (carry-on) - only process bags from index 1 onwards
        checked_bags = bag_list[1:] if len(bag_list) > 1 else []
        
        if not checked_bags:
            return base_price
        
        # Compute costs with optimal bag ordering (maximize complementary benefits)
        oversize_costs = [self._compute_oversize(bag, routine) for bag in checked_bags]
        overweight_if_comp = [
            self._compute_overweight(bag, routine, customer_class, complementary=True)
            for bag in checked_bags
        ]
        overweight_if_not_comp = [
            self._compute_overweight(bag, routine, customer_class, complementary=False)
            for bag in checked_bags
        ]
        
        # Calculate which bags benefit most from complementary status
        violation_if_comp = np.maximum(oversize_costs, overweight_if_comp)
        violation_if_not_comp = np.maximum(oversize_costs, overweight_if_not_comp)
        complementary_gain = np.array(violation_if_not_comp) - np.array(violation_if_comp)
        
        # Sort bags by complementary gain (descending) to maximize benefit
        order = np.argsort(-complementary_gain)
        sorted_bags = [checked_bags[i] for i in order]
        
        # Compute base check fees for sorted bags
        base_fees = self._compute_base_fees(sorted_bags, direction, routine, customer_class)
        
        # Determine which bags are complementary (base_fee == 0)
        complementary = [fee == 0 for fee in base_fees]
        
        # Recompute overweight with correct complementary status
        oversize_costs = [self._compute_oversize(bag, routine) for bag in sorted_bags]
        overweight_costs = [
            self._compute_overweight(bag, routine, customer_class, comp)
            for bag, comp in zip(sorted_bags, complementary)
        ]
        
        # Total violation cost (max of oversize, overweight per bag)
        violation_costs = np.maximum(oversize_costs, overweight_costs)
        
        total_baggage_fees = sum(base_fees) + sum(violation_costs)
        return base_price + int(total_baggage_fees)
    
    def _compute_base_fees(
        self,
        bag_list: List[Dict[str, Any]],
        direction: int,
        routine: str,
        customer_class: str,
    ) -> List[int]:
        """
        Compute base check fees for each bag.
        
        Args:
            bag_list: List of bags to check
            direction: 0 = outbound, 1 = return
            routine: Region/destination
            customer_class: Cabin class
        
        Returns:
            List of base fees for each bag.
        """
        base_fees = []
        
        for bag_idx, _ in enumerate(bag_list):
            # Bag number is capped at 4 (4th+ bag has same fee)
            table_idx = min(3, bag_idx)
            
            try:
                fee_table = self.fee_tables[table_idx][direction]
                if routine in fee_table.index and customer_class in fee_table.columns:
                    fee = fee_table.loc[routine, customer_class]
                    base_fees.append(int(fee))
                else:
                    # Fallback if region/class not found
                    print(f"Warning: No fee found for {routine}/{customer_class}")
                    base_fees.append(0)
            except Exception as e:
                print(f"Warning: Failed to lookup fee: {e}")
                base_fees.append(0)
        
        return base_fees
    
    def _compute_oversize(self, bag: Dict[str, Any], routine: str) -> int:
        """
        Compute oversize fee for a bag.
        
        Size is calculated as sum of dimensions (L + W + H).
        - <= 62 inches: No fee
        - 62-65 inches: $30
        - > 65 inches: $150 or $200 depending on region
        
        Args:
            bag: Bag dictionary with 'size' key [L, W, H]
            routine: Region/destination
        
        Returns:
            Oversize fee in USD.
        """
        size = bag.get("size", [0, 0, 0])
        total_size = sum(size)
        
        if total_size <= 62:
            return 0
        
        if total_size <= 65:
            return 30
        
        # Regions with $150 oversize fee (vs $200 for others)
        reduced_regions = [
            "Panama", "South America", "Peru", "Colombia", 
            "Ecuador", "Europe", "Israel", "Qatar"
        ]
        
        if routine in reduced_regions:
            return 150
        
        return 200
    
    def _compute_overweight(
        self, 
        bag: Dict[str, Any], 
        routine: str, 
        customer_class: str,
        complementary: bool
    ) -> int:
        """
        Compute overweight fee for a bag.
        
        Weight tiers:
        - <= 50 lbs: No fee
        - 50-53 lbs: $30
        - 53-70 lbs: $100 or $200 depending on region
        - 70-100 lbs: $200 or $450 depending on region
        
        First/Business class with complementary bags get 70 lbs free.
        
        Args:
            bag: Bag dictionary with 'weight' key
            routine: Region/destination
            customer_class: Cabin class
            complementary: Whether this bag is complementary
        
        Returns:
            Overweight fee in USD.
        """
        weight = bag.get("weight", 0)
        
        # Australia/New Zealand have special rules
        if routine in ["Australia", "New Zealand"]:
            if complementary:
                if weight <= 70:
                    return 0
                return 200
            if weight <= 50:
                return 0
            if weight <= 53:
                return 30
            if weight <= 70:
                return 100
            return 200
        
        # First/Business with complementary bags get 70 lbs allowance
        if complementary and customer_class in ["Business", "First"]:
            if weight <= 70:
                return 0
            # Over 70 lbs
            if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
                return 450
            return 200
        
        # Standard weight tiers
        if weight <= 50:
            return 0
        
        if weight <= 53:
            return 30
        
        if weight <= 70:
            if routine == "Cuba":
                return 200
            return 100
        
        # Over 70 lbs
        if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        
        return 200
    
    def load_rules(self) -> str:
        """
        Load the reference rules text file.
        
        Returns:
            Contents of reference_rules.txt as a string.
        
        Raises:
            FileNotFoundError: If the rules file doesn't exist.
        """
        rules_path = self.airline_path / "reference_rules.txt"
        
        if not rules_path.exists():
            # Try alternative filename
            alt_path = self.airline_path / "reference_rules_textual.txt"
            if alt_path.exists():
                rules_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Reference rules not found at: {rules_path}\n"
                    f"Make sure the RuleArena repository is properly cloned."
                )
        
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read rules file: {e}")
    
    def load_fee_table(self, bag_number: int, direction: int = 0) -> pd.DataFrame:
        """
        Load a specific fee lookup table.
        
        Args:
            bag_number: Bag number (1-4)
            direction: 0 = departing from U.S., 1 = arriving to U.S.
        
        Returns:
            DataFrame with fees indexed by region, columns by cabin class.
        
        Raises:
            ValueError: If bag_number is not 1-4 or direction is not 0-1.
        """
        if bag_number < 1 or bag_number > 4:
            raise ValueError(f"bag_number must be 1-4, got {bag_number}")
        
        if direction not in [0, 1]:
            raise ValueError(f"direction must be 0 or 1, got {direction}")
        
        table_idx = bag_number - 1
        return self.fee_tables[table_idx][direction].copy()
    
    def get_problem_count(self, complexity_level: int) -> int:
        """
        Get the number of problems at a given complexity level.
        
        Args:
            complexity_level: Problem complexity (0, 1, or 2)
        
        Returns:
            Number of problems in the JSONL file.
        """
        if complexity_level not in self.COMPLEXITY_FILES:
            return 0
        
        filename = self.COMPLEXITY_FILES[complexity_level]
        filepath = self.airline_path / "synthesized_problems" / filename
        
        if not filepath.exists():
            return 0
        
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        
        return count
    
    def get_all_regions(self) -> List[str]:
        """
        Get all unique regions from the fee tables.
        
        Returns:
            List of region names.
        """
        if not self.fee_tables or not self.fee_tables[0][0].index.any():
            return []
        
        return list(self.fee_tables[0][0].index)
    
    def get_cabin_classes(self) -> List[str]:
        """
        Get all cabin class names from the fee tables.
        
        Returns:
            List of cabin class names.
        """
        if not self.fee_tables or self.fee_tables[0][0].empty:
            return []
        
        return list(self.fee_tables[0][0].columns)


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("RuleArena Loader Test")
    print("=" * 60)
    
    try:
        loader = RuleArenaLoader("external/RuleArena")
        print("Loader initialized successfully")
        print()
        
        # Test loading problems
        print("Loading problems (complexity_level=0, max_problems=5)...")
        problems = loader.load_airline_problems(complexity_level=0, max_problems=5)
        print(f"Loaded {len(problems)} problems")
        print()
        
        # Show first problem
        if problems:
            print("First problem:")
            print("-" * 40)
            first = problems[0]
            print(f"  ID: {first['id']}")
            print(f"  Query: {first['query'][:100]}...")
            print(f"  Ground Truth: ${first['ground_truth']}")
            print(f"  Difficulty: {first['difficulty_level']}")
            print(f"  Passenger Context:")
            ctx = first['passenger_context']
            print(f"    - Class: {ctx['customer_class']}")
            print(f"    - Route: {ctx['routine']}")
            print(f"    - Direction: {'arriving' if ctx['direction'] else 'departing'}")
            print(f"    - Ticket: ${ctx['base_price']}")
            print(f"    - Bags: {ctx['num_bags']}")
            print()
        
        # Test loading rules
        print("Loading reference rules...")
        rules = loader.load_rules()
        print(f"Rules loaded: {len(rules)} characters")
        print(f"Preview: {rules[:200]}...")
        print()
        
        # Test fee table
        print("Loading fee table (bag 1, direction 0)...")
        fee_table = loader.load_fee_table(bag_number=1, direction=0)
        print(f"Fee table shape: {fee_table.shape}")
        print(f"Regions: {list(fee_table.index[:5])}...")
        print(f"Classes: {list(fee_table.columns)}")
        print()
        
        # Problem counts
        print("Problem counts by complexity:")
        for level in [0, 1, 2]:
            count = loader.get_problem_count(level)
            print(f"  Level {level}: {count} problems")
        
        print()
        print("All tests passed!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to clone the RuleArena repository first:")
        print("  git clone https://github.com/SkyRiver-2000/RuleArena external/RuleArena")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
