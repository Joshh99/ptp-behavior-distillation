"""
Aggregate RuleArena experiment results into summary CSV and JSON.

Scans benchmark_results/rulearena/ for result JSON files produced by run_single.py
and computes per-run summary statistics.

Usage:
    python -m benchmark.rulearena.aggregate_results
    python -m benchmark.rulearena.aggregate_results --run-id l1_ptool_debug
    python benchmark/rulearena/aggregate_results.py --help
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("benchmark_results") / "rulearena"

# Maps experiment name stems to levels (mirrors run_single.py)
LEVEL_MAP = {
    "l0_python": "L0",
    "l0f_cot": "L0F",
    "l1_ptool": "L1",
    "l1ta_tool_augmented": "L1-TA",
    "l3_react": "L3",
}

CSV_COLUMNS = [
    "run_id",
    "experiment_name",
    "level",
    "model",
    "domain",
    "complexity",
    "n",
    "accuracy_exact",
    "accuracy_tolerance",
    "f1_macro",
    "error_count",
    "error_rate",
    "cost_per_problem",
    "total_cost",
    "avg_latency_ms",
    "total_input_tokens",
    "total_output_tokens",
    "run_timestamp",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_run_id(filename_stem: str) -> Dict[str, str]:
    """
    Extract experiment name from filename stem.

    run_single.py writes files as: {experiment_name}_debug.json
    """
    name = filename_stem
    if name.endswith("_debug"):
        name = name[: -len("_debug")]
    return {"experiment_name": name}


def _infer_level(experiment_name: str) -> str:
    """Map experiment name to level string."""
    for prefix, level in LEVEL_MAP.items():
        if experiment_name.startswith(prefix):
            return level
    return "unknown"


def _compute_f1_macro(results: List[Dict]) -> Optional[float]:
    """
    Compute macro F1 for boolean classification (NBA domain only).

    Returns None if not applicable (non-boolean predictions).
    """
    # Check if all predictions are boolean
    preds = []
    labels = []
    for r in results:
        p = r.get("predicted")
        e = r.get("expected")
        if not isinstance(p, bool) or not isinstance(e, bool):
            return None
        preds.append(p)
        labels.append(e)

    if not preds:
        return None

    # Compute per-class precision/recall/f1, then macro average
    classes = [True, False]
    f1s = []
    for cls in classes:
        tp = sum(1 for p, l in zip(preds, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(preds, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(preds, labels) if p != cls and l == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        f1s.append(f1)

    return statistics.mean(f1s) if f1s else None


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_single_file(filepath: Path) -> Dict[str, Any]:
    """
    Aggregate a single result JSON file into a summary row.

    Args:
        filepath: Path to a JSON file containing a list of ExperimentResult dicts.

    Returns:
        Dict with one summary row matching CSV_COLUMNS.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle both envelope format {"run_summary": ..., "results": [...]}
    # and legacy flat list format [...]
    run_summary = None
    if isinstance(raw, dict) and "results" in raw:
        results = raw["results"]
        run_summary = raw.get("run_summary", {})
    elif isinstance(raw, list):
        results = raw
    else:
        return {}

    if not results:
        return {}

    run_id = filepath.stem
    parsed = _parse_run_id(run_id)
    experiment_name = parsed["experiment_name"]
    level = _infer_level(experiment_name)

    n = len(results)

    # Domain and complexity — take from metadata if consistent, else "mixed"
    domains = set()
    complexities = set()
    models = set()
    for r in results:
        md = r.get("metadata", {})
        domains.add(md.get("domain", "unknown"))
        complexities.add(str(md.get("complexity_level", "mixed")))
        # Model may be in metadata or top-level
        m = md.get("model") or r.get("model") or ""
        if m:
            models.add(m)

    domain = domains.pop() if len(domains) == 1 else "mixed"
    complexity = complexities.pop() if len(complexities) == 1 else "mixed"
    model = models.pop() if len(models) == 1 else "deepseek-ai/DeepSeek-V3"

    # Accuracy
    correct_exact = sum(
        1 for r in results
        if r.get("is_correct_exact") and not r.get("error")
    )
    correct_tolerance = sum(
        1 for r in results
        if r.get("is_correct_tolerance") and not r.get("error")
    )
    accuracy_exact = correct_exact / n if n > 0 else 0.0
    accuracy_tolerance = correct_tolerance / n if n > 0 else 0.0

    # F1 (NBA only) — prefer run_summary value, fall back to recomputation
    if run_summary and run_summary.get("f1_macro") is not None:
        f1_macro = run_summary["f1_macro"]
    else:
        nba_results = [r for r in results
                       if r.get("metadata", {}).get("domain") == "nba"]
        f1_macro = _compute_f1_macro(nba_results) if nba_results else None

    # Errors
    error_count = sum(1 for r in results if r.get("error"))
    error_rate = error_count / n if n > 0 else 0.0

    # Cost
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    cost_per_problem = total_cost / n if n > 0 else 0.0

    # Latency
    latencies = [r.get("latency_ms", 0.0) for r in results
                 if not r.get("error") and r.get("latency_ms", 0) > 0]
    avg_latency_ms = statistics.mean(latencies) if latencies else 0.0

    # Tokens
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)

    # Timestamp — use earliest instance timestamp, or file mtime
    timestamps = [r.get("timestamp", "") for r in results if r.get("timestamp")]
    if timestamps:
        run_timestamp = min(timestamps)
    else:
        mtime = filepath.stat().st_mtime
        run_timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "level": level,
        "model": model,
        "domain": domain,
        "complexity": complexity,
        "n": n,
        "accuracy_exact": round(accuracy_exact, 4),
        "accuracy_tolerance": round(accuracy_tolerance, 4),
        "f1_macro": round(f1_macro, 4) if f1_macro is not None else None,
        "error_count": error_count,
        "error_rate": round(error_rate, 4),
        "cost_per_problem": round(cost_per_problem, 6),
        "total_cost": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "run_timestamp": run_timestamp,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict], output_path: Path) -> None:
    """Write summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Wrote {}  ({} rows)".format(output_path, len(rows)))


def write_json(rows: List[Dict], output_path: Path) -> None:
    """Write summary rows to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("Wrote {}  ({} rows)".format(output_path, len(rows)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RuleArena experiment results into summary CSV/JSON."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Process a single result file by stem name (e.g. 'l1_ptool_debug'). "
             "Omit to process all JSON files in benchmark_results/rulearena/.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory containing result JSON files (default: benchmark_results/rulearena/)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="rq2_summary",
        help="Output filename prefix (default: rq2_summary)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print("Error: results directory not found: {}".format(results_dir))
        return

    # Collect files
    if args.run_id:
        # Single file mode
        candidates = [
            results_dir / "{}.json".format(args.run_id),
        ]
        files = [f for f in candidates if f.exists()]
        if not files:
            print("Error: no result file found for run-id '{}'".format(args.run_id))
            print("Looked for: {}".format(candidates))
            return
    else:
        # All files mode
        files = sorted(results_dir.glob("*.json"))
        # Exclude our own output files
        files = [
            f for f in files
            if not f.stem.startswith(args.output_prefix)
        ]

    if not files:
        print("No result files found in {}".format(results_dir))
        return

    print("Processing {} result file(s) from {}".format(len(files), results_dir))

    # Aggregate
    rows = []
    for filepath in files:
        try:
            row = aggregate_single_file(filepath)
            if row:
                rows.append(row)
                print("  {} -> n={}, exact={}, tol={}".format(
                    filepath.stem, row["n"],
                    row["accuracy_exact"], row["accuracy_tolerance"],
                ))
        except Exception as e:
            print("  {} -> ERROR: {}".format(filepath.stem, e))

    if not rows:
        print("No valid results to aggregate.")
        return

    # Sort by level then experiment name
    level_order = {"L0": 0, "L0F": 1, "L1": 2, "L1-TA": 3, "L3": 4, "unknown": 9}
    rows.sort(key=lambda r: (level_order.get(r["level"], 9), r["experiment_name"]))

    # Write outputs
    csv_path = results_dir / "{}.csv".format(args.output_prefix)
    json_path = results_dir / "{}.json".format(args.output_prefix)
    write_csv(rows, csv_path)
    write_json(rows, json_path)

    # Print summary table
    print("\n{:<30s} {:>5s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
        "Run", "N", "Exact", "Tol", "Cost", "Latency"))
    print("-" * 78)
    for row in rows:
        print("{:<30s} {:>5d} {:>7.1%} {:>7.1%} ${:>8.4f} {:>8.0f}ms".format(
            row["run_id"],
            row["n"],
            row["accuracy_exact"],
            row["accuracy_tolerance"],
            row["total_cost"],
            row["avg_latency_ms"],
        ))


if __name__ == "__main__":
    main()
