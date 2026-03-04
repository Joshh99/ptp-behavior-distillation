"""
Merge multiple JSON result files into a single consolidated file.

Features:
- Handles missing files gracefully
- Deduplicates results by (experiment, model, complexity_level)
- Preserves metadata from each source file
- Warns about inconsistencies (different problem counts, etc.)
"""
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_result_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a single result file, returning None if it doesn't exist or is invalid."""
    path = Path(filepath)
    
    if not path.exists():
        print(f"  WARNING: File not found: {filepath}")
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"  Loaded: {filepath} ({len(data.get('results', []))} experiments)")
        return data
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to read {filepath}: {e}")
        return None


def merge_results(
    input_files: List[str],
    output_file: str,
    deduplicate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Merge multiple result files into one.
    
    Args:
        input_files: List of JSON file paths to merge
        output_file: Output JSON file path
        deduplicate: If True, keep only the latest result for each (experiment, model, level)
        verbose: Print progress information
    
    Returns:
        Merged data dictionary
    """
    if verbose:
        print(f"\nMerging {len(input_files)} result files...")
        print("=" * 60)
    
    # Load all files
    loaded_data = []
    for filepath in input_files:
        data = load_result_file(filepath)
        if data is not None:
            loaded_data.append({
                'source': filepath,
                'data': data,
            })
    
    if not loaded_data:
        print("\nERROR: No valid files to merge!")
        return {}
    
    if verbose:
        print(f"\nSuccessfully loaded {len(loaded_data)}/{len(input_files)} files")
    
    # Merge results
    all_results = []
    source_files = []
    
    for item in loaded_data:
        source_files.append(item['source'])
        results = item['data'].get('results', [])
        
        for result in results:
            # Add source tracking
            result['_source_file'] = item['source']
            all_results.append(result)
    
    if verbose:
        print(f"Total experiments before deduplication: {len(all_results)}")
    
    # Deduplicate by (experiment, model, complexity_level)
    if deduplicate:
        seen = {}
        deduplicated = []
        duplicates = 0
        
        for result in all_results:
            key = (
                result.get('experiment'),
                result.get('model'),
                result.get('complexity_level'),
            )
            
            if key in seen:
                duplicates += 1
                # Keep the one with more problems, or the newer one
                existing = seen[key]
                if result.get('num_problems', 0) > existing.get('num_problems', 0):
                    seen[key] = result
            else:
                seen[key] = result
        
        deduplicated = list(seen.values())
        
        if verbose and duplicates > 0:
            print(f"Removed {duplicates} duplicate entries")
            print(f"Total experiments after deduplication: {len(deduplicated)}")
        
        all_results = deduplicated
    
    # Remove internal tracking field before saving
    for result in all_results:
        result.pop('_source_file', None)
    
    # Sort results for consistent output
    all_results.sort(key=lambda r: (
        r.get('complexity_level', 0),
        r.get('experiment', ''),
        r.get('model', ''),
    ))
    
    # Compute aggregate metadata
    total_problems = sum(r.get('num_problems', 0) for r in all_results)
    total_cost = sum(r.get('total_cost', 0) for r in all_results)
    
    models = sorted(set(r.get('model', 'unknown') for r in all_results))
    experiments = sorted(set(r.get('experiment', 'unknown') for r in all_results))
    levels = sorted(set(r.get('complexity_level', 0) for r in all_results))
    
    # Build merged output
    merged_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_files': source_files,
            'total_experiments': len(all_results),
            'total_problems': total_problems,
            'total_cost': total_cost,
            'models': models,
            'experiments': experiments,
            'complexity_levels': levels,
            'merged_from': len(loaded_data),
        },
        'results': all_results,
    }
    
    # Save merged file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    if verbose:
        print("\n" + "=" * 60)
        print("MERGE COMPLETE")
        print("=" * 60)
        print(f"Output: {output_file}")
        print(f"Experiments: {len(all_results)}")
        print(f"Models: {', '.join(models)}")
        print(f"Levels: {levels}")
        print(f"Total problems: {total_problems}")
        print(f"Total cost: ${total_cost:.4f}")
    
    return merged_data


def validate_results(data: Dict[str, Any]) -> List[str]:
    """Validate merged results and return list of warnings."""
    warnings = []
    results = data.get('results', [])
    
    if not results:
        warnings.append("No results found in merged data")
        return warnings
    
    # Check for incomplete experiments
    expected_experiments = {'l1_pure', 'l1_transparent', 'cot', 'tool_aug'}
    found_experiments = set(r.get('experiment') for r in results)
    missing = expected_experiments - found_experiments
    if missing:
        warnings.append(f"Missing experiments: {missing}")
    
    # Check for inconsistent problem counts
    problem_counts = {}
    for r in results:
        key = (r.get('model'), r.get('complexity_level'))
        count = r.get('num_problems', 0)
        if key not in problem_counts:
            problem_counts[key] = count
        elif problem_counts[key] != count:
            warnings.append(
                f"Inconsistent problem count for {key}: "
                f"{problem_counts[key]} vs {count}"
            )
    
    # Check for very low accuracy (potential issues)
    for r in results:
        acc = r.get('accuracy', 0)
        if acc < 0.1 and r.get('experiment') not in ['tool_aug']:
            warnings.append(
                f"Very low accuracy ({acc:.1%}) for "
                f"{r.get('experiment')} on level {r.get('complexity_level')}"
            )
    
    return warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple experiment result JSON files"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSON files to merge"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/merged_results.json",
        help="Output JSON file (default: results/merged_results.json)"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't remove duplicate experiments"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks on merged results"
    )
    
    args = parser.parse_args()
    
    merged = merge_results(
        input_files=args.input_files,
        output_file=args.output,
        deduplicate=not args.no_deduplicate,
    )
    
    if args.validate and merged:
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)
        warnings = validate_results(merged)
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")
        else:
            print("  All checks passed!")
