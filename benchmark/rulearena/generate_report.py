import json
import sys
from datetime import datetime
from pathlib import Path

from benchmark.rulearena.aggregate_results import aggregate_single_file


RESULTS_DIR = Path("benchmark_results/rulearena")
OUTPUT_FILE = Path("reports/rq2_report.html")

LEVEL_ORDER = {"L0": 0, "L0F": 1, "L1": 2, "L1-TA": 3, "L3": 4}

EXPERIMENT_DISPLAY = {
    "l0_python": "L0: Pure Python",
    "l0f_cot": "L0F: Chain-of-Thought",
    "l1_ptool": "L1: PTool Extraction",
    "l1ta_tool_augmented": "L1-TA: Tool-Augmented",
    "l3_react": "L3: ReAct Agent",
}

EXPERIMENT_COLORS = {
    "l0_python": "#7f8c8d",
    "l0f_cot": "#c0392b",
    "l1_ptool": "#2980b9",
    "l1ta_tool_augmented": "#f39c12",
    "l3_react": "#27ae60",
}


def load_rows(results_dir: Path):
    files = sorted(f for f in results_dir.glob("*.json") if not f.stem.startswith("rq2_summary"))
    rows = []
    for f in files:
        try:
            row = aggregate_single_file(f)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"Warning: could not process {f.name}: {e}")
    return rows


def generate_html(rows):
    if not rows:
        return _empty_report()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows.sort(key=lambda r: (
        LEVEL_ORDER.get(r.get("level", ""), 9),
        r.get("experiment_name", ""),
        r.get("domain", ""),
    ))

    total_n = sum(r.get("n", 0) for r in rows)
    best_acc = max((r.get("accuracy_tolerance", 0) for r in rows), default=0)
    total_cost = sum(r.get("total_cost", 0) for r in rows)
    num_runs = len(rows)

    summary_cards = f"""    <div id="summary" class="section">
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{total_n}</div>
                <div class="label">Total Problems</div>
            </div>
            <div class="summary-card">
                <div class="value">{num_runs}</div>
                <div class="label">Experiment Runs</div>
            </div>
            <div class="summary-card">
                <div class="value">{best_acc*100:.0f}%</div>
                <div class="label">Best Accuracy</div>
            </div>
            <div class="summary-card">
                <div class="value">${total_cost:.3f}</div>
                <div class="label">Total Cost</div>
            </div>
        </div>
    </div>"""

    table_rows = []
    for r in rows:
        exp = r.get("experiment_name", "")
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        level = r.get("level", "")
        domain = r.get("domain", "")
        n = r.get("n", 0)
        acc_tol = r.get("accuracy_tolerance", 0) * 100
        acc_exact = r.get("accuracy_exact", 0) * 100
        f1 = r.get("f1_macro")
        cost = r.get("total_cost", 0)
        latency = r.get("avg_latency_ms", 0)
        errors = r.get("error_count", 0)

        acc_class = "accuracy-high" if acc_tol >= 70 else "accuracy-medium" if acc_tol >= 40 else "accuracy-low"
        f1_str = f"{f1*100:.1f}%" if f1 is not None else "—"

        table_rows.append(f"""                <tr>
                    <td class="experiment-cell">{display}</td>
                    <td class="numeric">{level}</td>
                    <td class="numeric">{domain}</td>
                    <td class="numeric">{n}</td>
                    <td class="numeric {acc_class}">{acc_tol:.1f}%</td>
                    <td class="numeric">{acc_exact:.1f}%</td>
                    <td class="numeric">{f1_str}</td>
                    <td class="numeric">${cost:.4f}</td>
                    <td class="numeric">{latency:.0f}ms</td>
                    <td class="numeric">{errors}</td>
                </tr>""")

    results_table = f"""    <div id="results" class="section">
        <h2>Results by Experiment</h2>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Experiment</th>
                    <th rowspan="2">Level</th>
                    <th rowspan="2">Domain</th>
                    <th rowspan="2">N</th>
                    <th colspan="3" class="metric-header">Performance</th>
                    <th colspan="3" class="metric-header">Efficiency</th>
                </tr>
                <tr>
                    <th class="metric-header">Acc (tol)</th>
                    <th class="metric-header">Acc (exact)</th>
                    <th class="metric-header">F1 Macro</th>
                    <th class="metric-header">Cost</th>
                    <th class="metric-header">Latency</th>
                    <th class="metric-header">Errors</th>
                </tr>
            </thead>
            <tbody>
{chr(10).join(table_rows)}
            </tbody>
        </table>
        <p class="note">
            Acc (tol): within 1% relative tolerance.
            F1 Macro reported for NBA domain due to class imbalance (82.9% class imbalance).
        </p>
    </div>"""

    unique_exps = {}
    for r in rows:
        exp = r.get("experiment_name", "")
        if exp not in unique_exps:
            unique_exps[exp] = {"tol": [], "cost": []}
        unique_exps[exp]["tol"].append(r.get("accuracy_tolerance", 0))
        unique_exps[exp]["cost"].append(r.get("total_cost", 0))

    chart_exps = list(unique_exps.keys())
    chart_names = [EXPERIMENT_DISPLAY.get(e, e) for e in chart_exps]
    chart_acc = [sum(v["tol"]) / len(v["tol"]) * 100 for v in unique_exps.values()]
    chart_cost = [sum(v["cost"]) for v in unique_exps.values()]
    chart_colors = [EXPERIMENT_COLORS.get(e, "#666") for e in chart_exps]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQ2 Results — Behavior Distillation</title>
    {_styles()}
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
<div class="container">
    <header>
        <h1>RQ2 Results — Reliability/Autonomy Spectrum</h1>
        <div class="subtitle">RuleArena Benchmark: Airline, Tax, and NBA Domains</div>
        <div class="research-question">
            What is the optimal point on the reliability-generality-cost curve for production deployment?
        </div>
        <div class="timestamp">Generated: {timestamp}</div>
    </header>

    <nav>
        <a href="#summary">Summary</a>
        <a href="#results">Results Table</a>
        <a href="#visualizations">Charts</a>
        <a href="#methodology">Methodology</a>
    </nav>

    {summary_cards}

    {results_table}

    <div id="visualizations" class="section">
        <h2>Visualizations</h2>
        <div class="chart-container"><div id="accuracy-chart"></div></div>
        <div class="chart-container"><div id="cost-chart"></div></div>
        <div class="chart-container"><div id="efficiency-chart"></div></div>
    </div>

    <div id="methodology" class="section">
        <h2>Methodology</h2>
        <h3>Experiment Levels</h3>
        <ul>
            <li><strong>L0: Pure Python</strong> — Deterministic calculation, oracle parameters (ceiling)</li>
            <li><strong>L0F: Chain-of-Thought</strong> — Direct LLM reasoning, no structure (baseline)</li>
            <li><strong>L1: PTool Extraction</strong> — Structured extraction + deterministic calculation</li>
            <li><strong>L1-TA: Tool-Augmented</strong> — LLM generates and executes Python code</li>
            <li><strong>L3: ReAct Agent</strong> — Autonomous multi-step reasoning with tools</li>
        </ul>
        <h3>Domains</h3>
        <ul>
            <li><strong>Airline (300 problems)</strong> — Baggage fee computation across 3 complexity levels</li>
            <li><strong>Tax (300 problems)</strong> — Federal tax liability computation</li>
            <li><strong>NBA (216 problems)</strong> — Transaction compliance classification; F1 macro used due to 82.9% class imbalance</li>
        </ul>
        <h3>Model</h3>
        <p>deepseek-ai/DeepSeek-V3 via Together.ai — temperature 0.0, seed 42.</p>
    </div>

    <footer>
        <p>
            <strong>Behavior Distillation Research Project</strong><br>
            Independent Study with Prof. William Cohen (CMU Machine Learning / Google DeepMind)
        </p>
    </footer>
</div>

<script>
const expNames = {json.dumps(chart_names)};
const accuracies = {json.dumps(chart_acc)};
const costs = {json.dumps(chart_cost)};
const colors = {json.dumps(chart_colors)};

const layout = {{
    paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    font: {{ color: '#2c3e50', family: "'Palatino Linotype', Georgia, serif", size: 12 }},
    margin: {{ t: 50, b: 80, l: 60, r: 30 }}
}};

Plotly.newPlot('accuracy-chart', [{{
    x: expNames, y: accuracies, type: 'bar',
    marker: {{ color: colors }},
    text: accuracies.map(a => `${{a.toFixed(1)}}%`),
    textposition: 'outside'
}}], {{
    ...layout,
    title: {{ text: 'Average Accuracy (tolerance) by Experiment', font: {{ size: 16 }} }},
    yaxis: {{ title: 'Accuracy (%)', range: [0, Math.max(110, ...accuracies) + 10] }},
    height: 380
}});

Plotly.newPlot('cost-chart', [{{
    x: expNames, y: costs, type: 'bar',
    marker: {{ color: colors }},
    text: costs.map(c => `$${{c.toFixed(4)}}`),
    textposition: 'outside'
}}], {{
    ...layout,
    title: {{ text: 'Total Cost by Experiment (all runs)', font: {{ size: 16 }} }},
    yaxis: {{ title: 'Cost (USD)' }},
    height: 380
}});

const scatterTraces = expNames.map((name, i) => ({{
    x: [costs[i]], y: [accuracies[i]],
    mode: 'markers+text', type: 'scatter', name: name,
    marker: {{ size: 18, color: colors[i], line: {{ color: '#fff', width: 2 }} }},
    text: [name.split(':')[0]], textposition: 'top center',
    textfont: {{ color: '#2c3e50', size: 11 }}
}}));

Plotly.newPlot('efficiency-chart', scatterTraces, {{
    ...layout,
    title: {{ text: 'Cost vs Accuracy (Efficiency Frontier)', font: {{ size: 16 }} }},
    xaxis: {{ title: 'Total Cost (USD)' }},
    yaxis: {{ title: 'Accuracy (%)', range: [-5, Math.max(110, ...accuracies) + 10] }},
    showlegend: true,
    legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }},
    height: 420
}});
</script>
</body>
</html>
"""


def _empty_report():
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>RQ2 Report</title></head>
<body>
<h1>RQ2 Report</h1>
<p>No results found in <code>benchmark_results/rulearena/</code>.</p>
<p>Run experiments first:</p>
<pre>python -m benchmark.rulearena.run_single --experiment l1_ptool --domain airline --n 300 --seed 42</pre>
</body></html>
"""


def _styles():
    return """    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif;
            margin: 0; padding: 40px 20px;
            background: #fefdfb; color: #1a1a1a; line-height: 1.65;
        }
        .container { max-width: 1300px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 40px; padding: 30px 20px; border-bottom: 3px double #2c3e50; }
        h1 { font-size: 2.2em; font-weight: 600; margin-bottom: 8px; color: #2c3e50; }
        .subtitle { font-size: 1.15em; color: #34495e; }
        .research-question { font-style: italic; font-size: 0.95em; max-width: 750px; margin: 15px auto 0; color: #555; }
        .timestamp { font-size: 0.85em; color: #7f8c8d; margin-top: 12px; }
        nav { display: flex; gap: 8px; justify-content: center; margin: 30px 0; flex-wrap: wrap; }
        nav a { padding: 10px 18px; background: #ecf0f1; color: #2c3e50; text-decoration: none; border: 1px solid #bdc3c7; font-weight: 500; }
        nav a:hover { background: #d5dbdb; }
        .section { margin: 40px 0; padding: 20px 0; }
        h2 { font-size: 1.5em; font-weight: 600; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; margin-bottom: 20px; color: #2c3e50; }
        h3 { font-size: 1.2em; font-weight: 600; margin: 25px 0 15px; color: #34495e; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin: 20px 0; }
        .summary-card { border: 1px solid #dee2e6; padding: 20px 15px; text-align: center; background: #fff; border-radius: 4px; }
        .summary-card .value { font-size: 2.2em; font-weight: 600; color: #2c3e50; }
        .summary-card .label { font-size: 0.8em; margin-top: 5px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; border: 1px solid #dee2e6; font-size: 0.9em; }
        th, td { padding: 12px 10px; text-align: left; border: 1px solid #dee2e6; }
        th { background: #f8f9fa; font-weight: 600; text-align: center; color: #495057; }
        .metric-header { font-size: 0.8em; text-align: center; }
        td.numeric { text-align: center; font-family: 'Consolas', 'Monaco', monospace; }
        tr:hover { background: #f8f9fa; }
        .experiment-cell { font-weight: 600; white-space: nowrap; color: #2c3e50; }
        .accuracy-high { color: #27ae60; font-weight: 600; }
        .accuracy-medium { color: #f39c12; }
        .accuracy-low { color: #e74c3c; }
        .chart-container { margin: 30px 0; padding: 25px; border: 1px solid #dee2e6; background: #fff; border-radius: 4px; }
        .note { font-size: 0.85em; color: #6c757d; font-style: italic; margin-top: 10px; }
        footer { text-align: center; padding: 30px 20px; margin-top: 60px; border-top: 2px solid #dee2e6; font-size: 0.85em; color: #6c757d; }
        @media print { body { background: white; } nav { display: none; } }
    </style>"""


def main():
    if not RESULTS_DIR.exists():
        print(f"No results directory found at {RESULTS_DIR}")
        rows = []
    else:
        rows = load_rows(RESULTS_DIR)
        print(f"Loaded {len(rows)} result files from {RESULTS_DIR}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(rows)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
