"""
Report Generator for RuleArena Experiments

Generates self-contained HTML reports with Plotly visualizations.
Inspired by report.html and generate_html_results.py patterns.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from benchmark.rulearena.metrics.aggregator import AggregatedMetrics


class ReportGenerator:
    """Generate HTML reports with visualizations from aggregated metrics."""

    EXPERIMENT_DISPLAY_NAMES = {
        'l0_python': 'L0: Pure Python',
        'l0f_cot': 'L0F: Chain-of-Thought',
        'l1_ptool': 'L1: PTool Extraction',
        'l1ta_tool_augmented': 'L1-TA: Tool-Augmented',
        'l3_react': 'L3: ReAct Agent',
    }

    EXPERIMENT_COLORS = {
        'l0_python': '#7f8c8d',
        'l0f_cot': '#c0392b',
        'l1_ptool': '#2980b9',
        'l1ta_tool_augmented': '#f39c12',
        'l3_react': '#27ae60',
    }

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save report and metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        all_metrics: Dict[str, AggregatedMetrics],
        model_id: str = "deepseek-ai/DeepSeek-V3",
        seed: int = 42,
    ):
        """
        Generate complete report with metrics.json and report.html.

        Args:
            all_metrics: Dict mapping experiment names to AggregatedMetrics
            model_id: Model identifier used in experiments
            seed: Random seed used
        """
        # Save metrics.json
        metrics_file = self.output_dir / "metrics.json"
        metrics_data = {
            name: metrics.to_dict()
            for name, metrics in all_metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Saved metrics.json to {metrics_file}")

        # Generate report.html
        html = self._generate_html(all_metrics, model_id, seed)
        report_file = self.output_dir / "report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Saved report.html to {report_file}")

    def _generate_html(
        self,
        all_metrics: Dict[str, AggregatedMetrics],
        model_id: str,
        seed: int,
    ) -> str:
        """Generate complete HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RuleArena RQ2 Results - Behavior Distillation Study</title>
    {self._get_styles()}
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
<div class="container">
    <header>
        <h1>RuleArena Research Question 2 Results</h1>
        <div class="subtitle">Reliability-Autonomy Spectrum Experiments</div>
        <div class="research-question">
            What is the optimal point on the reliability-generality-cost curve for production deployment?
        </div>
        <div class="timestamp">Generated: {timestamp}</div>
        <div class="metadata">Model: {model_id} | Seed: {seed}</div>
    </header>

    <nav>
        <a href="#summary">Summary</a>
        <a href="#results">Results Table</a>
        <a href="#visualizations">Visualizations</a>
        <a href="#findings">Key Findings</a>
        <a href="#methodology">Methodology</a>
    </nav>

    {self._generate_summary_section(all_metrics)}
    {self._generate_results_table(all_metrics)}
    {self._generate_visualizations(all_metrics)}
    {self._generate_findings(all_metrics)}
    {self._generate_methodology()}

    <footer>
        <p>
            <strong>Behavior Distillation Research Project</strong><br>
            Independent Study with Prof. William Cohen (CMU Machine Learning / Google DeepMind)<br>
            Student: Joshua Wisdom Momo | Spring 2026
        </p>
    </footer>
</div>

{self._generate_scripts(all_metrics)}
</body>
</html>
"""
        return html

    def _get_styles(self) -> str:
        """Return CSS styles for the report."""
        return """    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif;
            margin: 0;
            padding: 40px 20px;
            background: #fefdfb;
            color: #1a1a1a;
            line-height: 1.65;
        }
        .container { max-width: 1300px; margin: 0 auto; }

        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 20px;
            border-bottom: 3px double #2c3e50;
        }
        h1 {
            font-size: 2.2em;
            font-weight: 600;
            margin-bottom: 8px;
            color: #2c3e50;
            letter-spacing: -0.5px;
        }
        .subtitle {
            font-size: 1.15em;
            margin-bottom: 5px;
            color: #34495e;
        }
        .research-question {
            font-style: italic;
            font-size: 0.95em;
            max-width: 750px;
            margin: 15px auto 0;
            color: #555;
        }
        .timestamp, .metadata {
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 12px;
        }

        nav {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        nav a {
            padding: 10px 18px;
            background: #ecf0f1;
            color: #2c3e50;
            text-decoration: none;
            border: 1px solid #bdc3c7;
            font-weight: 500;
            transition: all 0.2s;
        }
        nav a:hover {
            background: #d5dbdb;
            border-color: #95a5a6;
        }

        .section {
            margin: 40px 0;
            padding: 20px 0;
        }
        h2 {
            font-size: 1.5em;
            font-weight: 600;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 8px;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        h3 {
            font-size: 1.2em;
            font-weight: 600;
            margin: 25px 0 15px 0;
            color: #34495e;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .summary-card {
            border: 1px solid #dee2e6;
            padding: 20px 15px;
            text-align: center;
            background: #fff;
            border-radius: 4px;
        }
        .summary-card .value {
            font-size: 2.2em;
            font-weight: 600;
            color: #2c3e50;
        }
        .summary-card .label {
            font-size: 0.8em;
            margin-top: 5px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border: 1px solid #dee2e6;
            font-size: 0.9em;
        }
        th, td {
            padding: 12px 10px;
            text-align: left;
            border: 1px solid #dee2e6;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            text-align: center;
            color: #495057;
        }
        .metric-header {
            font-size: 0.8em;
            text-align: center;
        }
        td.numeric {
            text-align: center;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        tr:hover {
            background: #f8f9fa;
        }

        .experiment-cell {
            font-weight: 600;
            white-space: nowrap;
            color: #2c3e50;
        }

        .accuracy-high { color: #27ae60; font-weight: 600; }
        .accuracy-medium { color: #f39c12; }
        .accuracy-low { color: #e74c3c; }

        .chart-container {
            margin: 30px 0;
            padding: 25px;
            border: 1px solid #dee2e6;
            background: #fff;
            border-radius: 4px;
        }

        .insight-box {
            background: #fff;
            border-left: 4px solid #2c3e50;
            padding: 20px 25px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .insight-box h4 {
            margin: 0 0 12px 0;
            font-size: 1.05em;
            font-weight: 600;
            color: #2c3e50;
        }
        .insight-box p {
            margin: 8px 0;
            color: #495057;
        }

        .note {
            font-size: 0.85em;
            color: #6c757d;
            font-style: italic;
            margin-top: 10px;
        }

        footer {
            text-align: center;
            padding: 30px 20px;
            margin-top: 60px;
            border-top: 2px solid #dee2e6;
            font-size: 0.85em;
            color: #6c757d;
        }

        @media print {
            body { background: white; }
            nav { display: none; }
        }
    </style>"""

    def _generate_summary_section(self, all_metrics: Dict[str, AggregatedMetrics]) -> str:
        """Generate executive summary cards."""
        if not all_metrics:
            return ""

        total_instances = sum(m.total_instances for m in all_metrics.values())
        total_experiments = len(all_metrics)
        best_accuracy = max(m.accuracy_tolerance for m in all_metrics.values()) * 100
        total_cost = sum(m.total_cost_usd for m in all_metrics.values())
        avg_latency = sum(m.avg_latency_ms for m in all_metrics.values()) / len(all_metrics) / 1000

        return f"""    <div id="summary" class="section">
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{total_instances}</div>
                <div class="label">Total Problems</div>
            </div>
            <div class="summary-card">
                <div class="value">{total_experiments}</div>
                <div class="label">Experiments Run</div>
            </div>
            <div class="summary-card">
                <div class="value">{best_accuracy:.1f}%</div>
                <div class="label">Best Accuracy</div>
            </div>
            <div class="summary-card">
                <div class="value">${total_cost:.3f}</div>
                <div class="label">Total Cost</div>
            </div>
            <div class="summary-card">
                <div class="value">{avg_latency:.1f}s</div>
                <div class="label">Avg Latency</div>
            </div>
        </div>
    </div>"""

    def _generate_results_table(self, all_metrics: Dict[str, AggregatedMetrics]) -> str:
        """Generate comparison table of all experiments."""
        if not all_metrics:
            return ""

        rows = []
        for exp_name, metrics in sorted(all_metrics.items(), key=lambda x: x[1].accuracy_tolerance, reverse=True):
            display_name = self.EXPERIMENT_DISPLAY_NAMES.get(exp_name, exp_name)
            acc = metrics.accuracy_tolerance * 100
            acc_exact = metrics.accuracy_exact * 100

            acc_class = 'accuracy-high' if acc >= 70 else 'accuracy-medium' if acc >= 40 else 'accuracy-low'

            rows.append(f"""                <tr>
                    <td class="experiment-cell">{display_name}</td>
                    <td class="numeric">{metrics.experiment_level}</td>
                    <td class="numeric">{metrics.total_instances}</td>
                    <td class="numeric {acc_class}">{acc:.1f}%</td>
                    <td class="numeric">{acc_exact:.1f}%</td>
                    <td class="numeric">{metrics.correct_tolerance}/{metrics.total_instances}</td>
                    <td class="numeric">${metrics.total_cost_usd:.4f}</td>
                    <td class="numeric">${metrics.avg_cost_usd:.6f}</td>
                    <td class="numeric">{metrics.avg_latency_ms/1000:.1f}s</td>
                    <td class="numeric">{metrics.error_count}</td>
                </tr>""")

        return f"""    <div id="results" class="section">
        <h2>Results by Experiment</h2>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Experiment</th>
                    <th rowspan="2">Level</th>
                    <th rowspan="2">N</th>
                    <th colspan="3" class="metric-header">Performance</th>
                    <th colspan="3" class="metric-header">Efficiency</th>
                    <th rowspan="2">Errors</th>
                </tr>
                <tr>
                    <th class="metric-header">Acc (tol)</th>
                    <th class="metric-header">Acc (exact)</th>
                    <th class="metric-header">Correct</th>
                    <th class="metric-header">Total Cost</th>
                    <th class="metric-header">Avg Cost</th>
                    <th class="metric-header">Avg Time</th>
                </tr>
            </thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
        </table>
    </div>"""

    def _generate_visualizations(self, all_metrics: Dict[str, AggregatedMetrics]) -> str:
        """Generate placeholder for Plotly charts."""
        return """    <div id="visualizations" class="section">
        <h2>Visualizations</h2>

        <div class="chart-container">
            <div id="accuracy-chart"></div>
        </div>

        <div class="chart-container">
            <div id="cost-chart"></div>
        </div>

        <div class="chart-container">
            <div id="efficiency-chart"></div>
        </div>

        <div class="chart-container">
            <div id="domain-breakdown"></div>
        </div>
    </div>"""

    def _generate_findings(self, all_metrics: Dict[str, AggregatedMetrics]) -> str:
        """Generate key findings based on results."""
        if not all_metrics:
            return ""

        sorted_by_acc = sorted(all_metrics.items(), key=lambda x: x[1].accuracy_tolerance, reverse=True)
        best_exp, best_metrics = sorted_by_acc[0]
        worst_exp, worst_metrics = sorted_by_acc[-1] if len(sorted_by_acc) > 1 else (None, None)

        best_name = self.EXPERIMENT_DISPLAY_NAMES.get(best_exp, best_exp)

        findings = f"""    <div id="findings" class="section">
        <h2>Key Findings</h2>

        <div class="insight-box">
            <h4>Best Performer: {best_name}</h4>
            <p>
                Achieved <strong>{best_metrics.accuracy_tolerance*100:.1f}% accuracy</strong>
                on {best_metrics.total_instances} problems.
                Total cost: ${best_metrics.total_cost_usd:.4f}
                (${best_metrics.avg_cost_usd:.6f} per problem).
            </p>
        </div>"""

        if worst_metrics and worst_exp != best_exp:
            worst_name = self.EXPERIMENT_DISPLAY_NAMES.get(worst_exp, worst_exp)
            acc_diff = (best_metrics.accuracy_tolerance - worst_metrics.accuracy_tolerance) * 100
            cost_ratio = best_metrics.avg_cost_usd / worst_metrics.avg_cost_usd if worst_metrics.avg_cost_usd > 0 else 0

            findings += f"""

        <div class="insight-box">
            <h4>Approach Comparison</h4>
            <p>
                <strong>{best_name}</strong> outperforms <strong>{worst_name}</strong> by
                {acc_diff:.1f} percentage points in accuracy.
                {f"Cost ratio: {cost_ratio:.2f}x." if cost_ratio > 0 else ""}
            </p>
        </div>"""

        # Add domain breakdown if available
        if best_metrics.by_category:
            findings += """

        <h3>Domain Breakdown</h3>"""
            for domain, cat_metrics in best_metrics.by_category.items():
                findings += f"""
        <div class="insight-box">
            <h4>{domain.capitalize()} Domain</h4>
            <p>
                Best accuracy: {cat_metrics.accuracy_tolerance*100:.1f}%
                ({cat_metrics.correct_tolerance}/{cat_metrics.total_instances} correct)
            </p>
        </div>"""

        findings += """
    </div>"""
        return findings

    def _generate_methodology(self) -> str:
        """Generate methodology section."""
        return """    <div id="methodology" class="section">
        <h2>Methodology</h2>

        <h3>Research Question</h3>
        <p>
            What is the optimal point on the reliability-generality-cost curve for production deployment
            of LLM-based systems on rule-guided reasoning tasks?
        </p>

        <h3>Experiment Levels</h3>
        <ul>
            <li><strong>L0: Pure Python</strong> - Deterministic calculation with oracle parameters (ceiling)</li>
            <li><strong>L0F: Chain-of-Thought</strong> - Direct LLM reasoning (baseline)</li>
            <li><strong>L1: PTool Extraction</strong> - Structured extraction + deterministic calculation (core hypothesis)</li>
            <li><strong>L1-TA: Tool-Augmented</strong> - LLM generates and executes code</li>
            <li><strong>L3: ReAct Agent</strong> - Autonomous multi-step reasoning with tools</li>
        </ul>

        <h3>Dataset</h3>
        <p>
            RuleArena benchmark (Zhou et al., ACL 2025) - 816 problems across 3 domains:
        </p>
        <ul>
            <li>Airline Baggage Fees (10 rules, 300 problems)</li>
            <li>NBA Transactions (54 rules, 216 problems)</li>
            <li>Tax Regulations (31 rules, 300 problems)</li>
        </ul>
        <p>
            Each domain has 3 complexity levels (0: simple, 1: medium, 2: complex).
        </p>

        <h3>Model Configuration</h3>
        <ul>
            <li><strong>Model:</strong> deepseek-ai/DeepSeek-V3 via Together.ai</li>
            <li><strong>Temperature:</strong> 0.0 (deterministic)</li>
            <li><strong>Seed:</strong> 42</li>
            <li><strong>Pricing:</strong> $0.30/M input tokens, $0.90/M output tokens</li>
        </ul>

        <h3>Metrics</h3>
        <ul>
            <li><strong>Accuracy (exact):</strong> Predicted answer matches expected (rounded to 2 decimal places)</li>
            <li><strong>Accuracy (tolerance):</strong> Within 1% relative tolerance</li>
            <li><strong>Cost:</strong> Total API cost in USD</li>
            <li><strong>Latency:</strong> Wall-clock time per problem</li>
            <li><strong>Error rate:</strong> Percentage of instances that failed to complete</li>
        </ul>
    </div>"""

    def _generate_scripts(self, all_metrics: Dict[str, AggregatedMetrics]) -> str:
        """Generate JavaScript for interactive charts."""
        if not all_metrics:
            return ""

        # Prepare data for charts
        exp_names = []
        exp_display_names = []
        accuracies = []
        costs = []
        colors = []

        for exp_name, metrics in sorted(all_metrics.items(), key=lambda x: x[1].accuracy_tolerance, reverse=True):
            exp_names.append(exp_name)
            exp_display_names.append(self.EXPERIMENT_DISPLAY_NAMES.get(exp_name, exp_name))
            accuracies.append(metrics.accuracy_tolerance * 100)
            costs.append(metrics.total_cost_usd)
            colors.append(self.EXPERIMENT_COLORS.get(exp_name, '#666'))

        return f"""<script>
// Chart data
const expNames = {json.dumps(exp_display_names)};
const accuracies = {json.dumps(accuracies)};
const costs = {json.dumps(costs)};
const colors = {json.dumps(colors)};

const layout = {{
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    font: {{ color: '#2c3e50', family: "'Palatino Linotype', Georgia, serif", size: 12 }},
    xaxis: {{ gridcolor: '#ecf0f1', zerolinecolor: '#ecf0f1' }},
    yaxis: {{ gridcolor: '#ecf0f1', zerolinecolor: '#ecf0f1' }},
    margin: {{ t: 50, b: 80, l: 60, r: 30 }}
}};

// Accuracy chart
Plotly.newPlot('accuracy-chart', [{{
    x: expNames,
    y: accuracies,
    type: 'bar',
    marker: {{ color: colors, line: {{ color: '#fff', width: 1 }} }},
    text: accuracies.map(a => `${{a.toFixed(1)}}%`),
    textposition: 'outside',
    textfont: {{ color: '#2c3e50', size: 13 }}
}}], {{
    ...layout,
    title: {{ text: 'Accuracy by Experiment (Tolerance)', font: {{ size: 16, color: '#2c3e50' }} }},
    yaxis: {{ ...layout.yaxis, title: 'Accuracy (%)', range: [0, Math.max(110, ...accuracies) + 10] }},
    height: 380
}});

// Cost chart
Plotly.newPlot('cost-chart', [{{
    x: expNames,
    y: costs,
    type: 'bar',
    marker: {{ color: colors, line: {{ color: '#fff', width: 1 }} }},
    text: costs.map(c => `$${{c.toFixed(4)}}`),
    textposition: 'outside',
    textfont: {{ color: '#2c3e50', size: 13 }}
}}], {{
    ...layout,
    title: {{ text: 'Total Cost by Experiment', font: {{ size: 16, color: '#2c3e50' }} }},
    yaxis: {{ ...layout.yaxis, title: 'Cost (USD)' }},
    height: 380
}});

// Efficiency scatter (Cost vs Accuracy)
const scatterTraces = expNames.map((name, i) => ({{
    x: [costs[i]],
    y: [accuracies[i]],
    mode: 'markers+text',
    type: 'scatter',
    name: name,
    marker: {{ size: 18, color: colors[i], line: {{ color: '#fff', width: 2 }} }},
    text: [name.split(':')[0]],
    textposition: 'top center',
    textfont: {{ color: '#2c3e50', size: 11 }}
}}));

Plotly.newPlot('efficiency-chart', scatterTraces, {{
    ...layout,
    title: {{ text: 'Cost vs Accuracy (Efficiency Frontier)', font: {{ size: 16, color: '#2c3e50' }} }},
    xaxis: {{ ...layout.xaxis, title: 'Total Cost (USD)' }},
    yaxis: {{ ...layout.yaxis, title: 'Accuracy (%)', range: [-5, Math.max(110, ...accuracies) + 10] }},
    showlegend: true,
    legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }},
    height: 420
}});

// Domain breakdown (if available)
const metricsData = {json.dumps({name: m.to_dict() for name, m in all_metrics.items()})};
const firstExp = Object.values(metricsData)[0];
if (firstExp && firstExp.by_category) {{
    const domains = Object.keys(firstExp.by_category);
    const domainTraces = Object.entries(metricsData).map(([expName, metrics]) => ({{
        x: domains,
        y: domains.map(d => (metrics.by_category[d]?.accuracy_tolerance || 0) * 100),
        name: expNames[Object.keys(metricsData).indexOf(expName)],
        type: 'bar'
    }}));

    Plotly.newPlot('domain-breakdown', domainTraces, {{
        ...layout,
        title: {{ text: 'Accuracy by Domain', font: {{ size: 16, color: '#2c3e50' }} }},
        yaxis: {{ ...layout.yaxis, title: 'Accuracy (%)' }},
        barmode: 'group',
        height: 400
    }});
}}
</script>"""
