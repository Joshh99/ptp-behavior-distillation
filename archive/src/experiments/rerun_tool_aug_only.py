"""
Re-run ONLY tool_aug experiments for DeepSeek and Qwen.
Merges new results with existing L1/CoT results.

This saves ~$18 by not re-running L1 and CoT experiments.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from src.dataset.airline_loader import AirlineLoader
from src.experiments.tool_aug_baseline import run_tool_augmented
from src.experiments.run_full_baseline import run_experiment, NumpyEncoder


def rerun_tool_aug_and_merge(
    existing_results_file: str,
    output_file: str,
    model: str,
    num_problems: int = 100,
    complexity_levels: list = [0, 1, 2],
):
    """
    Load existing results, re-run only tool_aug, merge and save.
    
    Args:
        existing_results_file: Path to existing JSON with L1/CoT results
        output_file: Where to save merged results
        model: Model to use for tool_aug
        num_problems: Problems per level
        complexity_levels: Which levels to run
    """
    print("=" * 70)
    print(f"RE-RUNNING TOOL_AUG ONLY")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Levels: {complexity_levels}")
    print(f"Problems per level: {num_problems}")
    print(f"Existing results: {existing_results_file}")
    print(f"Output: {output_file}")
    print("=" * 70)
    
    # Load existing results
    existing_path = Path(existing_results_file)
    if not existing_path.exists():
        print(f"ERROR: {existing_results_file} not found!")
        return
    
    with open(existing_path, 'r') as f:
        existing_data = json.load(f)
    
    existing_results = existing_data.get('results', [])
    print(f"\nLoaded {len(existing_results)} existing experiments")
    
    # Filter out old tool_aug results (we'll replace them)
    kept_results = []
    removed_count = 0
    for r in existing_results:
        if r.get('experiment') == 'tool_aug' and r.get('model') == model:
            removed_count += 1
            print(f"  Removing old: tool_aug level {r.get('complexity_level')}")
        else:
            kept_results.append(r)
    
    print(f"Kept {len(kept_results)} experiments (removed {removed_count} old tool_aug)")
    
    # Run new tool_aug experiments
    loader = AirlineLoader("external/RuleArena")
    
    new_tool_aug_results = []
    for level in complexity_levels:
        print(f"\n{'='*70}")
        print(f"Running tool_aug for level {level}...")
        print(f"{'='*70}")
        
        result = run_experiment(
            experiment_name="tool_aug",
            experiment_fn=run_tool_augmented,
            loader=loader,
            complexity_level=level,
            num_problems=num_problems,
            model=model,
            verbose=True,
        )
        new_tool_aug_results.append(result)
        
        print(f"\nLevel {level} complete: {result['accuracy']*100:.1f}% accuracy")
    
    # Merge results
    all_results = kept_results + new_tool_aug_results
    
    # Sort by level, then experiment name
    all_results.sort(key=lambda r: (
        r.get('complexity_level', 0),
        r.get('experiment', ''),
    ))
    
    # Calculate totals
    total_cost = sum(r.get('total_cost', 0) for r in all_results)
    
    # Save merged results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "note": "tool_aug re-run with fixed numpy type handling",
            "complexity_levels": complexity_levels,
            "num_problems_per_level": num_problems,
        },
        "results": all_results,
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - MERGED RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<20} {'Level':<7} {'Accuracy':<12} {'Cost':<12}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['experiment']:<20} "
              f"{r['complexity_level']:<7} "
              f"{r['accuracy']*100:>6.1f}%     "
              f"${r['total_cost']:>9.4f}")
    
    print("-" * 70)
    print(f"Total cost: ${total_cost:.2f}")
    print(f"\nSaved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Re-run only tool_aug and merge with existing results"
    )
    parser.add_argument(
        "--existing",
        required=True,
        help="Path to existing results JSON file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for merged results"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=100,
        help="Problems per level (default: 100)"
    )
    parser.add_argument(
        "--complexity",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Complexity levels (default: 0 1 2)"
    )
    
    args = parser.parse_args()
    
    rerun_tool_aug_and_merge(
        existing_results_file=args.existing,
        output_file=args.output,
        model=args.model,
        num_problems=args.num_problems,
        complexity_levels=args.complexity,
    )
