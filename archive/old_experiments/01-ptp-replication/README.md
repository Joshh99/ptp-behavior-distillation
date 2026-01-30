# PTP Core Replication

## Objectives
- Implement basic ptools using secretagent
- Replicate sports_understanding task from PTP paper
- Validate ptool-based reasoning vs direct LLM calls

## Experiments
- `sports_understanding.py` - Core implementation
- `baselines.py` - Direct LLM comparison

## Results
See `results/` folder for outputs and analysis.
EOF

cat > experiments/02-workflow-spectrum/README.md << 'EOF'
# Workflow Complexity Spectrum (L0-L3)

## Objectives
- Map L0 (fixed) → L1 (router) → L2 (state graph) → L3 (agentic)
- Implement multiple complexity levels for same task
- Quantify reliability vs generality tradeoff

## Experiments
- `l0_fixed_workflow.py` - Deterministic pipeline
- `l1_router.py` - LLM-based routing
- `l2_state_graph.py` - LangGraph implementation
- `l3_agentic.py` - ReAct-style autonomous agent

## Results
Comparative analysis in `results/spectrum_comparison.md`