
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_files": summary["num_files"],
        "num_physical_groups": summary["num_physical_groups"],
        "num_evaluations": summary["num_evaluations"],
        "accuracy_strict": summary["accuracy_strict"],
        "accuracy_operational": summary["accuracy_operational"],
        "mean_regret": summary["mean_regret"],
        "median_regret": summary["median_regret"],
        "p95_regret": summary["p95_regret"],
        "max_regret": summary["max_regret"],
    }


def _agreement(baseline: dict[str, Any], fine_tuned: dict[str, Any]) -> float:
    def actions(summary: dict[str, Any]) -> dict[tuple[str, int], int]:
        return {
            (item["filename"], int(item["shuffle"])): int(item["suggested_source_idx"])
            for item in summary["detailed_results"]
        }

    before, after = actions(baseline), actions(fine_tuned)
    if before.keys() != after.keys():
        raise ValueError("Valutazioni baseline e fine-tuned non allineate")
    return 100.0 * sum(before[key] == after[key] for key in before) / len(before)


def evaluate(args: argparse.Namespace) -> Path:
    from recommender.rlfuzzy import evaluate_model

    baseline = evaluate_model(
        args.data, args.baseline, seed=args.seed, n_shuffles=args.shuffles,
        verbose=False, show_solution_distances=False,
    )
    fine_tuned = evaluate_model(
        args.data, args.model, seed=args.seed, n_shuffles=args.shuffles,
        verbose=False, show_solution_distances=False,
    )
    baseline_compact = _compact(baseline)
    fine_tuned_compact = _compact(fine_tuned)
    report = {
        "data": str(Path(args.data).resolve()),
        "baseline_model": str(Path(args.baseline).resolve()),
        "fine_tuned_model": str(Path(args.model).resolve()),
        "evaluation_seed": args.seed,
        "shuffles": args.shuffles,
        "top1_agreement_with_baseline": _agreement(baseline, fine_tuned),
        "baseline": baseline_compact,
        "fine_tuned": fine_tuned_compact,
        "delta": {
            "accuracy_strict_pp": (
                fine_tuned_compact["accuracy_strict"] - baseline_compact["accuracy_strict"]
            ),
            "accuracy_operational_pp": (
                fine_tuned_compact["accuracy_operational"]
                - baseline_compact["accuracy_operational"]
            ),
            "mean_regret": fine_tuned_compact["mean_regret"] - baseline_compact["mean_regret"],
        },
    }
    output = Path(args.output) if args.output else Path(args.model).resolve().parent / "test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("model        accuracy_strict  accuracy_operational  mean_regret")
    print(
        f"baseline     {baseline_compact['accuracy_strict']:15.2f}  "
        f"{baseline_compact['accuracy_operational']:20.2f}  {baseline_compact['mean_regret']:11.4f}"
    )
    print(
        f"fine_tuned   {fine_tuned_compact['accuracy_strict']:15.2f}  "
        f"{fine_tuned_compact['accuracy_operational']:20.2f}  {fine_tuned_compact['mean_regret']:11.4f}"
    )
    print(
        f"delta        {report['delta']['accuracy_strict_pp']:+15.2f}  "
        f"{report['delta']['accuracy_operational_pp']:+20.2f}  "
        f"{report['delta']['mean_regret']:+11.4f}"
    )
    print(f"Accordo top-1 col PPO originale: {report['top1_agreement_with_baseline']:.2f}%")
    print(f"Report: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default="best_model/best_model/best_model.zip")
    parser.add_argument("--data", default="data/addestramento/test")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.shuffles <= 0:
        parser.error("--shuffles deve essere positivo")
    try:
        evaluate(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

