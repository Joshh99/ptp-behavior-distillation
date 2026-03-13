import json
import argparse
from pathlib import Path
from benchmark.rulearena.dataset.loader import RuleArenaDataset
from benchmark.rulearena.experiments.l3_react import L3_ReAct_Experiment


def retry_instances(domain, instance_ids, results_path):
    ds = RuleArenaDataset()
    target_ids = set(instance_ids)
    instances = [i for i in ds.instances if i.instance_id in target_ids]

    if not instances:
        print(f"No instances found for {target_ids}")
        return

    print(f"Retrying {len(instances)} instances: {[i.instance_id for i in instances]}")

    results_file = Path(results_path)
    existing = json.loads(results_file.read_text())
    existing_results = existing["results"]

    exp = L3_ReAct_Experiment()
    new_results = []
    for inst in instances:
        result = exp.run_instance(inst)
        new_results.append(result.__dict__ if hasattr(result, '__dict__') else result)
        print(f"[{inst.instance_id}] predicted={result.predicted}, correct={result.is_correct_exact}")

    # Replace failed entries with new results
    id_to_new = {r["instance_id"]: r for r in new_results}
    updated = []
    replaced = 0
    for r in existing_results:
        if r["instance_id"] in id_to_new:
            updated.append(id_to_new[r["instance_id"]])
            replaced += 1
        else:
            updated.append(r)

    existing["results"] = updated
    results_file.write_text(json.dumps(existing, indent=2, default=str))
    print(f"Replaced {replaced} entries in {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    retry_instances(args.domain, args.ids, args.results)