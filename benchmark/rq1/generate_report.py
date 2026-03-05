import json
import sys
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("results/rq1")
OUTPUT_FILE = Path("reports/rq1_report.html")

EXPERIMENT_DISPLAY = {
    "l1_pure": "L1 PTool (Pure)",
    "l1_transparent": "L1 PTool (Transparent)",
    "cot": "CoT Baseline",
    "tool_aug": "Tool-Augmented",
}

EXPERIMENT_COLORS = {
    "l1_pure": "#2980b9",
    "l1_transparent": "#1abc9c",
    "cot": "#c0392b",
    "tool_aug": "#f39c12",
}


def load_results(results_dir: Path):
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            results = data.get("results", []) if isinstance(data, dict) else data
            rows.extend(results)
        except Exception as e:
            print(f"Warning: could not load {f}: {e}")
    return rows


def generate_html(rows):
    if not rows:
        return _empty_report()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_experiments = len(rows)
    total_problems = sum(r.get("num_problems", 0) for r in rows)
    best_accuracy = max((r.get("accuracy", 0) for r in rows), default=0)
    total_cost = sum(r.get("total_cost", 0) for r in rows)

    models = sorted(set(r.get("model", "") for r in rows if r.get("model")))
    model_label = models[0] if len(models) == 1 else f"{len(models)} models"

    summary_cards = f"""    <div id="summary" class="section">
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{total_problems}</div>
                <div class="label">Total Problems</div>
            </div>
            <div class="summary-card">
                <div class="value">{total_experiments}</div>
                <div class="label">Experiment Runs</div>
            </div>
            <div class="summary-card">
                <div class="value">{best_accuracy*100:.0f}%</div>
                <div class="label">Best Accuracy</div>
            </div>
            <div class="summary-card">
                <div class="value">${total_cost:.3f}</div>
                <div class="label">Total Cost</div>
            </div>
        </div>
    </div>"""

    table_rows = []
    for r in sorted(rows, key=lambda x: (-x.get("accuracy", 0), x.get("experiment", ""))):
        exp = r.get("experiment", "")
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        level = r.get("complexity_level", "")
        model = r.get("model", "")
        n = r.get("num_problems", 0)
        acc = r.get("accuracy", 0) * 100
        avg_cost = r.get("avg_cost", 0)
        avg_time = r.get("avg_time", 0)
        acc_class = "accuracy-high" if acc >= 70 else "accuracy-medium" if acc >= 40 else "accuracy-low"
        table_rows.append(f"""                <tr>
                    <td class="experiment-cell">{display}</td>
                    <td class="numeric">{level}</td>
                    <td class="numeric">{n}</td>
                    <td class="numeric {acc_class}">{acc:.1f}%</td>
                    <td class="numeric">${avg_cost:.6f}</td>
                    <td class="numeric">{avg_time:.2f}s</td>
                    <td class="numeric" style="font-size:0.8em">{model}</td>
                </tr>""")

    results_table = f"""    <div id="results" class="section">
        <h2>Results by Experiment</h2>
        <table>
            <thead>
                <tr>
                    <th>Experiment</th>
                    <th>Level</th>
                    <th>N</th>
                    <th>Accuracy</th>
                    <th>Avg Cost</th>
                    <th>Avg Time</th>
                    <th>Model</th>
                </tr>
            </thead>
            <tbody>
{chr(10).join(table_rows)}
            </tbody>
        </table>
        <p class="note">Accuracy is exact match (predicted == expected dollar amount).</p>
    </div>"""

    exp_names_js = json.dumps([
        EXPERIMENT_DISPLAY.get(r.get("experiment", ""), r.get("experiment", ""))
        for r in rows
    ])
    accuracies_js = json.dumps([r.get("accuracy", 0) * 100 for r in rows])
    costs_js = json.dumps([r.get("total_cost", 0) for r in rows])
    colors_js = json.dumps([
        EXPERIMENT_COLORS.get(r.get("experiment", ""), "#666") for r in rows
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQ1 Results — Behavior Distillation</title>
    {_styles()}
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
<div class="container">
    <header>
        <h1>RQ1 Baseline Results</h1>
        <div class="subtitle">Airline Baggage Fee Domain — Reliability/Autonomy Spectrum</div>
        <div class="research-question">
            How do L1 PTool approaches compare to CoT and Tool-Augmented baselines on rule-guided reasoning?
        </div>
        <div class="timestamp">Generated: {timestamp} | Model: {model_label}</div>
    </header>

    <nav>
        <a href="#summary">Summary</a>
        <a href="#results">Results Table</a>
        <a href="#visualizations">Charts</a>
    </nav>

    {summary_cards}

    {results_table}

    <div id="visualizations" class="section">
        <h2>Visualizations</h2>
        <div class="chart-container"><div id="accuracy-chart"></div></div>
        <div class="chart-container"><div id="cost-chart"></div></div>
    </div>

    <footer>
        <p>
            <strong>Behavior Distillation Research Project</strong><br>
            Independent Study with Prof. William Cohen (CMU Machine Learning / Google DeepMind)
        </p>
    </footer>
</div>

<script>
const expNames = {exp_names_js};
const accuracies = {accuracies_js};
const costs = {costs_js};
const colors = {colors_js};

const layout = {{
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
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
    title: {{ text: 'Accuracy by Experiment', font: {{ size: 16 }} }},
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
    title: {{ text: 'Total Cost by Experiment', font: {{ size: 16 }} }},
    yaxis: {{ title: 'Cost (USD)' }},
    height: 380
}});
</script>
</body>
</html>
"""


def _empty_report():
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>RQ1 Report</title></head>
<body>
<h1>RQ1 Report</h1>
<p>No results found in <code>results/rq1/</code>.</p>
<p>Run <code>python -m benchmark.rq1.run_full_baseline</code> to generate results.</p>
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
        .container { max-width: 1200px; margin: 0 auto; }
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
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin: 20px 0; }
        .summary-card { border: 1px solid #dee2e6; padding: 20px 15px; text-align: center; background: #fff; border-radius: 4px; }
        .summary-card .value { font-size: 2.2em; font-weight: 600; color: #2c3e50; }
        .summary-card .label { font-size: 0.8em; margin-top: 5px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; border: 1px solid #dee2e6; font-size: 0.9em; }
        th, td { padding: 12px 10px; text-align: left; border: 1px solid #dee2e6; }
        th { background: #f8f9fa; font-weight: 600; text-align: center; color: #495057; }
        td.numeric { text-align: center; font-family: 'Consolas', 'Monaco', monospace; }
        tr:hover { background: #f8f9fa; }
        .experiment-cell { font-weight: 600; color: #2c3e50; }
        .accuracy-high { color: #27ae60; font-weight: 600; }
        .accuracy-medium { color: #f39c12; }
        .accuracy-low { color: #e74c3c; }
        .chart-container { margin: 30px 0; padding: 25px; border: 1px solid #dee2e6; background: #fff; border-radius: 4px; }
        .note { font-size: 0.85em; color: #6c757d; font-style: italic; margin-top: 10px; }
        footer { text-align: center; padding: 30px 20px; margin-top: 60px; border-top: 2px solid #dee2e6; font-size: 0.85em; color: #6c757d; }
    </style>"""


def main():
    results_dir = RESULTS_DIR
    output_file = OUTPUT_FILE

    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        print("Run experiments first: python -m benchmark.rq1.run_full_baseline")
        rows = []
    else:
        rows = load_results(results_dir)
        print(f"Loaded {len(rows)} experiment runs from {results_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(rows)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written to {output_file}")


if __name__ == "__main__":
    main()
