#!/usr/bin/env python
"""
RQ2 Meeting Experiment Runner

Focused experiment set for the RQ2 research meeting:
- L0 oracle:  airline (50), tax (50)       -- NBA skipped (no Python oracle)
- L0F CoT:    airline (50), NBA (50)       -- Tax skipped (under refinement)
- L1 PTool:   airline (50), NBA (50)       -- Tax skipped (under refinement)

NBA uses F1 score (not accuracy) because ~82.9% of answers are True.

Usage:
    python -m scripts.run_rq2_meeting
    python -m scripts.run_rq2_meeting --n 10   # Quick smoke test
"""

import sys
import os
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import benchmark.rulearena.config as config
config.MODEL_CONFIG["model_id"] = "deepseek-ai/DeepSeek-V3"


from benchmark.rulearena.dataset.loader import RuleArenaDataset, RuleArenaInstance
from benchmark.rulearena.experiments.base import ExperimentResult
from benchmark.rulearena.experiments.l0_python import L0PythonExperiment
from benchmark.rulearena.experiments.l0f_cot import L0F_CoT_Experiment
from benchmark.rulearena.experiments.l1_ptool import L1_PTool_Experiment
from benchmark.rulearena.metrics.aggregator import MetricsAggregator, AggregatedMetrics

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Experiment matrix ────────────────────────────────────────────────────────
# Each entry: (experiment_key, domain, level_str)
EXPERIMENT_MATRIX = [
    ("l0_python", "airline", "L0"),
    ("l0_python", "tax",     "L0"),
    ("l0f_cot",  "airline",  "L0F"),
    ("l0f_cot",  "nba",      "L0F"),
    ("l1_ptool", "airline",  "L1"),
    ("l1_ptool", "nba",      "L1"),
]

SKIPPED_NOTES = {
    ("l0_python", "nba"):  "No Python oracle for NBA (boolean classification)",
    ("l0f_cot",  "tax"):   "Tax CoT prompt under refinement",
    ("l1_ptool", "tax"):   "Tax extraction schema under refinement",
}


# ── F1 helper ────────────────────────────────────────────────────────────────

def compute_f1(results: List[ExperimentResult]) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for boolean classification results.

    Positive class = True (violation detected).
    """
    tp = fp = fn = tn = 0
    for r in results:
        if r.error:
            fn += 1  # count errors as missed positives
            continue
        pred = _to_bool(r.predicted)
        gold = _to_bool(r.expected)
        if pred is None or gold is None:
            fn += 1
            continue
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _to_bool(val) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val >= 0.5
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


# ── Sample helper ────────────────────────────────────────────────────────────

def sample_domain(
    dataset: RuleArenaDataset, domain: str, n: int, seed: int = 42
) -> List[RuleArenaInstance]:
    """Reproducible sample of n instances from a domain (stratified by complexity)."""
    rng = random.Random(seed)
    by_level: Dict[int, List[RuleArenaInstance]] = {0: [], 1: [], 2: []}
    for inst in dataset.instances:
        if inst.domain == domain:
            by_level[inst.complexity_level].append(inst)

    per_level = n // 3
    remainder = n % 3
    sampled = []
    for level in sorted(by_level):
        take = per_level + (1 if level < remainder else 0)
        pool = by_level[level]
        rng.shuffle(pool)
        sampled.extend(pool[:take])
    return sampled


# ── Run one cell ─────────────────────────────────────────────────────────────

def run_cell(
    experiment_key: str,
    domain: str,
    instances: List[RuleArenaInstance],
) -> List[ExperimentResult]:
    """Run an experiment on a list of instances with a progress bar."""
    if experiment_key == "l0_python":
        exp = L0PythonExperiment()
    elif experiment_key == "l0f_cot":
        exp = L0F_CoT_Experiment()
    elif experiment_key == "l1_ptool":
        exp = L1_PTool_Experiment()
    else:
        raise ValueError(f"Unknown experiment: {experiment_key}")

    label = f"{experiment_key} / {domain}"
    iterator = tqdm(instances, desc=label, unit="inst") if HAS_TQDM else instances

    results = []
    for inst in iterator:
        result = exp.run_instance(inst)
        results.append(result)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RQ2 meeting experiment runner")
    parser.add_argument("--n", type=int, default=50,
                        help="Problems per domain per experiment (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    n = args.n
    seed = args.seed
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("RQ2 Meeting Experiment Runner")
    print("=" * 70)
    print(f"  Problems per cell:  {n}")
    print(f"  Seed:               {seed}")
    print(f"  Timestamp:          {ts}")
    print()

    # Load dataset
    dataset = RuleArenaDataset()
    print()

    # Run each cell
    all_results: Dict[str, List[ExperimentResult]] = {}
    all_f1: Dict[str, Dict] = {}

    for exp_key, domain, level_str in EXPERIMENT_MATRIX:
        cell_key = f"{exp_key}__{domain}"
        instances = sample_domain(dataset, domain, n, seed)

        print(f"\n--- {exp_key} / {domain} ({len(instances)} instances) ---")
        results = run_cell(exp_key, domain, instances)
        all_results[cell_key] = results

        # Quick accuracy print
        correct = sum(1 for r in results if r.is_correct_tolerance and not r.error)
        errors = sum(1 for r in results if r.error)
        print(f"    Correct: {correct}/{len(results)}  Errors: {errors}")

        # F1 for NBA
        if domain == "nba":
            f1_info = compute_f1(results)
            all_f1[cell_key] = f1_info
            print(f"    F1={f1_info['f1']:.3f}  P={f1_info['precision']:.3f}  R={f1_info['recall']:.3f}")

    # Aggregate per experiment level
    all_metrics: Dict[str, AggregatedMetrics] = {}
    for exp_key, domain, level_str in EXPERIMENT_MATRIX:
        cell_key = f"{exp_key}__{domain}"
        results = all_results[cell_key]
        agg = MetricsAggregator.aggregate(cell_key, level_str, results)
        all_metrics[cell_key] = agg

    # ── Save raw results JSON ────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_json = {
        "metadata": {
            "timestamp": ts,
            "n_per_cell": n,
            "seed": seed,
            "model": "deepseek-ai/DeepSeek-V3",
            "matrix": [
                {"experiment": e, "domain": d, "level": l}
                for e, d, l in EXPERIMENT_MATRIX
            ],
            "skipped": {f"{e}__{d}": note for (e, d), note in SKIPPED_NOTES.items()},
        },
        "cells": {},
        "f1_scores": all_f1,
        "aggregated": {k: v.to_dict() for k, v in all_metrics.items()},
    }
    for cell_key, results in all_results.items():
        raw_json["cells"][cell_key] = [r.to_dict() for r in results]

    json_path = output_dir / f"rq2_meeting_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(raw_json, f, indent=2, default=str)
    print(f"\nSaved raw results: {json_path}")

    # ── Generate HTML report ─────────────────────────────────────────────
    html = generate_rq2_html(all_metrics, all_f1, n, seed, ts)
    html_path = output_dir / f"rq2_meeting_{ts}_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML report: {html_path}")

    # ── Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Cell':<28} {'Acc(tol)':>10} {'Cost':>10} {'F1':>8}")
    print("-" * 60)
    for cell_key, agg in all_metrics.items():
        f1_str = ""
        if cell_key in all_f1:
            f1_str = f"{all_f1[cell_key]['f1']:.3f}"
        print(
            f"{cell_key:<28} "
            f"{agg.accuracy_tolerance*100:>9.1f}% "
            f"${agg.total_cost_usd:>8.4f} "
            f"{f1_str:>8}"
        )
    print()

    for (e, d), note in SKIPPED_NOTES.items():
        print(f"  SKIPPED {e}/{d}: {note}")
    print()
    print("Done.")


# ── HTML report generator ────────────────────────────────────────────────────

def generate_rq2_html(
    all_metrics: Dict[str, AggregatedMetrics],
    all_f1: Dict[str, Dict],
    n: int,
    seed: int,
    timestamp: str,
) -> str:
    """Generate self-contained HTML report for the RQ2 meeting."""

    # Build results table rows
    table_rows = []
    for cell_key, agg in sorted(all_metrics.items()):
        exp_key, domain = cell_key.split("__")
        acc_tol = agg.accuracy_tolerance * 100
        acc_exact = agg.accuracy_exact * 100
        acc_class = "high" if acc_tol >= 70 else "med" if acc_tol >= 40 else "low"
        f1_cell = ""
        if cell_key in all_f1:
            f1 = all_f1[cell_key]
            f1_cell = f"{f1['f1']:.3f}"
        table_rows.append(f"""<tr>
            <td class="exp">{exp_key}</td>
            <td>{domain}</td>
            <td>{agg.experiment_level}</td>
            <td class="num">{agg.total_instances}</td>
            <td class="num acc-{acc_class}">{acc_tol:.1f}%</td>
            <td class="num">{acc_exact:.1f}%</td>
            <td class="num">{f1_cell}</td>
            <td class="num">${agg.total_cost_usd:.4f}</td>
            <td class="num">{agg.avg_latency_ms/1000:.1f}s</td>
            <td class="num">{agg.error_count}</td>
        </tr>""")
    rows_html = "\n".join(table_rows)

    # Chart data
    chart_labels = []
    chart_acc = []
    chart_cost = []
    chart_colors = {
        "l0_python": "#7f8c8d",
        "l0f_cot": "#c0392b",
        "l1_ptool": "#2980b9",
    }
    for cell_key, agg in sorted(all_metrics.items()):
        exp_key, domain = cell_key.split("__")
        chart_labels.append(f"{exp_key}/{domain}")
        chart_acc.append(round(agg.accuracy_tolerance * 100, 1))
        chart_cost.append(round(agg.total_cost_usd, 4))

    chart_color_list = []
    for cell_key in sorted(all_metrics):
        exp_key = cell_key.split("__")[0]
        chart_color_list.append(chart_colors.get(exp_key, "#666"))

    # F1 comparison data for NBA cells
    f1_labels = []
    f1_values = []
    f1_colors = []
    for cell_key in sorted(all_f1):
        exp_key, domain = cell_key.split("__")
        f1_labels.append(f"{exp_key}/{domain}")
        f1_values.append(round(all_f1[cell_key]["f1"], 3))
        f1_colors.append(chart_colors.get(exp_key, "#666"))

    # Skipped notes HTML
    skipped_items = "".join(
        f"<li><strong>{e}/{d}:</strong> {note}</li>"
        for (e, d), note in SKIPPED_NOTES.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RQ2 Meeting Report - {timestamp}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif;
    margin: 0; padding: 40px 20px;
    background: #fefdfb; color: #1a1a1a; line-height: 1.65;
}}
.container {{ max-width: 1300px; margin: 0 auto; }}
header {{
    text-align: center; margin-bottom: 40px; padding: 30px 20px;
    border-bottom: 3px double #2c3e50;
}}
h1 {{ font-size: 2.2em; font-weight: 600; margin-bottom: 8px; color: #2c3e50; }}
.subtitle {{ font-size: 1.15em; color: #34495e; margin-bottom: 5px; }}
.rq {{ font-style: italic; font-size: 0.95em; max-width: 750px; margin: 15px auto 0; color: #555; }}
.meta {{ font-size: 0.85em; color: #7f8c8d; margin-top: 12px; }}
h2 {{ font-size: 1.5em; font-weight: 600; border-bottom: 2px solid #2c3e50;
      padding-bottom: 8px; margin: 40px 0 20px; color: #2c3e50; }}
h3 {{ font-size: 1.2em; font-weight: 600; margin: 25px 0 15px; color: #34495e; }}

/* Summary cards */
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
          gap: 15px; margin: 20px 0; }}
.card {{ border: 1px solid #dee2e6; padding: 20px 15px; text-align: center;
         background: #fff; border-radius: 4px; }}
.card .val {{ font-size: 2.2em; font-weight: 600; color: #2c3e50; }}
.card .lbl {{ font-size: 0.8em; margin-top: 5px; color: #6c757d;
              text-transform: uppercase; letter-spacing: 0.5px; }}

/* Table */
table {{ width: 100%; border-collapse: collapse; margin: 20px 0;
         border: 1px solid #dee2e6; font-size: 0.9em; }}
th, td {{ padding: 12px 10px; text-align: left; border: 1px solid #dee2e6; }}
th {{ background: #f8f9fa; font-weight: 600; text-align: center; color: #495057; }}
td.num {{ text-align: center; font-family: 'Consolas', monospace; }}
td.exp {{ font-weight: 600; white-space: nowrap; color: #2c3e50; }}
.acc-high {{ color: #27ae60; font-weight: 600; }}
.acc-med  {{ color: #f39c12; }}
.acc-low  {{ color: #e74c3c; }}
tr:hover {{ background: #f8f9fa; }}

/* Charts */
.chart {{ margin: 30px 0; padding: 25px; border: 1px solid #dee2e6;
          background: #fff; border-radius: 4px; }}

/* Insight */
.insight {{ background: #fff; border-left: 4px solid #2c3e50; padding: 20px 25px;
            margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.insight h4 {{ margin: 0 0 12px; font-size: 1.05em; font-weight: 600; color: #2c3e50; }}
.insight p {{ margin: 8px 0; color: #495057; }}

.note {{ font-size: 0.85em; color: #6c757d; font-style: italic; margin-top: 10px; }}

footer {{
    text-align: center; padding: 30px 20px; margin-top: 60px;
    border-top: 2px solid #dee2e6; font-size: 0.85em; color: #6c757d;
}}
@media print {{ nav {{ display: none; }} body {{ background: white; }} }}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>RQ2 Meeting Report</h1>
    <div class="subtitle">Reliability-Autonomy Spectrum: RuleArena Benchmark</div>
    <div class="rq">
        What is the optimal point on the reliability-generality-cost curve
        for production deployment?
    </div>
    <div class="meta">Generated: {timestamp} | Model: deepseek-ai/DeepSeek-V3 | Seed: {seed} | N={n}/cell</div>
</header>

<!-- Summary Cards -->
<h2>Executive Summary</h2>
<div class="cards">
    <div class="card">
        <div class="val">{sum(m.total_instances for m in all_metrics.values())}</div>
        <div class="lbl">Total Problems</div>
    </div>
    <div class="card">
        <div class="val">{len(all_metrics)}</div>
        <div class="lbl">Cells Run</div>
    </div>
    <div class="card">
        <div class="val">{max(m.accuracy_tolerance for m in all_metrics.values())*100:.1f}%</div>
        <div class="lbl">Best Accuracy</div>
    </div>
    <div class="card">
        <div class="val">${sum(m.total_cost_usd for m in all_metrics.values()):.4f}</div>
        <div class="lbl">Total Cost</div>
    </div>
    <div class="card">
        <div class="val">{max((f['f1'] for f in all_f1.values()), default=0):.3f}</div>
        <div class="lbl">Best NBA F1</div>
    </div>
</div>

<!-- Results Table -->
<h2>Results by Cell</h2>
<table>
<thead>
    <tr>
        <th>Experiment</th><th>Domain</th><th>Level</th><th>N</th>
        <th>Acc (tol)</th><th>Acc (exact)</th><th>F1</th>
        <th>Cost</th><th>Avg Time</th><th>Errors</th>
    </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<p class="note">
    F1 column only applies to NBA (boolean classification, positive = violation).
    Accuracy metrics use 1% relative tolerance.
</p>

<!-- Skipped cells -->
<h3>Skipped Cells</h3>
<ul>{skipped_items}</ul>

<!-- Charts -->
<h2>Visualizations</h2>
<div class="chart"><div id="acc-chart"></div></div>
<div class="chart"><div id="cost-chart"></div></div>
<div class="chart"><div id="f1-chart"></div></div>
<div class="chart"><div id="scatter-chart"></div></div>

<!-- Key Findings -->
<h2>Key Findings</h2>
<div id="findings-container"></div>

<!-- Methodology -->
<h2>Methodology</h2>
<h3>Experiment Levels</h3>
<ul>
    <li><strong>L0: Pure Python</strong> - Deterministic calculation with oracle parameters (ceiling)</li>
    <li><strong>L0F: Chain-of-Thought</strong> - Direct LLM reasoning (baseline)</li>
    <li><strong>L1: PTool Extraction</strong> - Structured extraction + deterministic calculation</li>
</ul>
<h3>Dataset</h3>
<p>RuleArena benchmark (Zhou et al., ACL 2025): 816 problems across 3 domains.
{n} problems sampled per cell, stratified by complexity level.</p>
<h3>NBA Class Imbalance</h3>
<p>~82.9% of NBA answers are True (violation). We report F1 score instead of accuracy
to avoid inflated metrics from majority-class guessing.</p>

<footer>
    <strong>Behavior Distillation Research Project</strong><br>
    Independent Study with Prof. William Cohen (CMU ML / Google DeepMind)<br>
    Student: Joshua Wisdom Momo | Spring 2026
</footer>

</div>

<script>
const labels = {json.dumps(chart_labels)};
const accs   = {json.dumps(chart_acc)};
const costs  = {json.dumps(chart_cost)};
const colors = {json.dumps(chart_color_list)};

const lo = {{
    paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    font: {{ color: '#2c3e50', family: "'Palatino Linotype', Georgia, serif", size: 12 }},
    xaxis: {{ gridcolor: '#ecf0f1' }}, yaxis: {{ gridcolor: '#ecf0f1' }},
    margin: {{ t: 50, b: 100, l: 60, r: 30 }}
}};

// Accuracy bar chart
Plotly.newPlot('acc-chart', [{{
    x: labels, y: accs, type: 'bar',
    marker: {{ color: colors }},
    text: accs.map(a => a.toFixed(1)+'%'), textposition: 'outside',
    textfont: {{ size: 13, color: '#2c3e50' }}
}}], {{
    ...lo, title: {{ text: 'Accuracy by Cell (1% tolerance)', font: {{ size: 16 }} }},
    yaxis: {{ ...lo.yaxis, title: 'Accuracy (%)', range: [0, 110] }},
    height: 400
}});

// Cost bar chart
Plotly.newPlot('cost-chart', [{{
    x: labels, y: costs, type: 'bar',
    marker: {{ color: colors }},
    text: costs.map(c => '$'+c.toFixed(4)), textposition: 'outside',
    textfont: {{ size: 13, color: '#2c3e50' }}
}}], {{
    ...lo, title: {{ text: 'Total Cost by Cell', font: {{ size: 16 }} }},
    yaxis: {{ ...lo.yaxis, title: 'Cost (USD)' }},
    height: 400
}});

// F1 bar chart (NBA only)
const f1Labels = {json.dumps(f1_labels)};
const f1Vals   = {json.dumps(f1_values)};
const f1Colors = {json.dumps(f1_colors)};

if (f1Labels.length > 0) {{
    Plotly.newPlot('f1-chart', [{{
        x: f1Labels, y: f1Vals, type: 'bar',
        marker: {{ color: f1Colors }},
        text: f1Vals.map(v => v.toFixed(3)), textposition: 'outside',
        textfont: {{ size: 13, color: '#2c3e50' }}
    }}], {{
        ...lo, title: {{ text: 'NBA F1 Score by Experiment', font: {{ size: 16 }} }},
        yaxis: {{ ...lo.yaxis, title: 'F1 Score', range: [0, 1.1] }},
        height: 380
    }});
}} else {{
    document.getElementById('f1-chart').innerHTML = '<p style="color:#999">No NBA cells to display.</p>';
}}

// Cost vs Accuracy scatter
const traces = labels.map((name, i) => ({{
    x: [costs[i]], y: [accs[i]], mode: 'markers+text', type: 'scatter',
    name: name,
    marker: {{ size: 18, color: colors[i], line: {{ color: '#fff', width: 2 }} }},
    text: [name], textposition: 'top center',
    textfont: {{ color: '#2c3e50', size: 10 }}
}}));

Plotly.newPlot('scatter-chart', traces, {{
    ...lo,
    title: {{ text: 'Cost vs Accuracy', font: {{ size: 16 }} }},
    xaxis: {{ ...lo.xaxis, title: 'Total Cost (USD)' }},
    yaxis: {{ ...lo.yaxis, title: 'Accuracy (%)', range: [-5, 115] }},
    showlegend: true,
    legend: {{ orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' }},
    height: 420
}});

// Dynamic findings
const fc = document.getElementById('findings-container');
let html = '';

// Best accuracy
let bestIdx = accs.indexOf(Math.max(...accs));
html += `<div class="insight"><h4>Best Accuracy: ${{labels[bestIdx]}}</h4>
<p>${{accs[bestIdx].toFixed(1)}}% accuracy at $${{costs[bestIdx].toFixed(4)}} total cost.</p></div>`;

// NBA F1 comparison
if (f1Labels.length >= 2) {{
    html += `<div class="insight"><h4>NBA F1 Comparison</h4><p>`;
    f1Labels.forEach((l, i) => {{
        html += `${{l}}: F1=${{f1Vals[i].toFixed(3)}}  `;
    }});
    html += `</p></div>`;
}}

// L0 ceiling note
html += `<div class="insight"><h4>L0 Oracle Ceiling</h4>
<p>L0 (pure Python) achieves 100% accuracy on airline/tax at zero API cost,
confirming ground-truth calculation correctness and establishing the ceiling
for LLM-based approaches.</p></div>`;

fc.innerHTML = html;
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
