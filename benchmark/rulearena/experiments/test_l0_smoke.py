"""
Smoke test for L0 Python oracle experiment.

Runs 5 airline + 5 tax instances and prints per-instance results.
"""

from benchmark.rulearena.dataset.loader import RuleArenaDataset
from benchmark.rulearena.experiments.l0_python import L0PythonExperiment


def main():
    dataset = RuleArenaDataset()
    experiment = L0PythonExperiment()

    airline = [i for i in dataset.instances if i.domain == "airline"][:5]
    tax = [i for i in dataset.instances if i.domain == "tax"][:5]

    print(f"{'instance_id':<22} {'computed':>12} {'expected':>12}  status")
    print("-" * 65)

    for instance in airline + tax:
        result = experiment.run_instance(instance)
        if result.error:
            tag = f"ERROR: {result.error}"
        elif result.is_correct_exact:
            tag = "MATCH"
        else:
            tag = "FAIL"
        print(
            f"{result.instance_id:<22} "
            f"{str(result.predicted):>12} "
            f"{str(result.expected):>12}  "
            f"{tag}"
        )


if __name__ == "__main__":
    main()
