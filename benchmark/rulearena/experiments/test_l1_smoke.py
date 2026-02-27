"""
Smoke test for L1 PTool extraction experiment.

Runs 5 instances per domain and prints per-instance results.
Requires TOGETHER_API_KEY environment variable.
"""

from benchmark.rulearena.dataset.loader import RuleArenaDataset
from benchmark.rulearena.experiments.l1_ptool import L1_PTool_Experiment


def main():
    dataset = RuleArenaDataset()
    experiment = L1_PTool_Experiment()

    airline = [i for i in dataset.instances if i.domain == "airline"][:5]
    tax = [i for i in dataset.instances if i.domain == "tax"][:5]
    nba = [i for i in dataset.instances if i.domain == "nba"][:5]

    print(f"{'instance_id':<22} {'predicted':>14} {'expected':>14}  status")
    print("-" * 75)

    for instance in airline + tax + nba:
        result = experiment.run_instance(instance)
        if result.error:
            tag = f"ERROR: {result.error[:40]}"
        elif result.is_correct_exact:
            tag = "MATCH"
        elif result.is_correct_tolerance:
            tag = "TOLERANCE"
        else:
            tag = "FAIL"
        print(
            f"{result.instance_id:<22} "
            f"{str(result.predicted):>14} "
            f"{str(result.expected):>14}  "
            f"{tag}"
        )


if __name__ == "__main__":
    main()
