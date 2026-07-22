"""Final train/validation/test comparison for one MaskablePPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.rlfuzzy import compare_train_validation_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronta lo stesso checkpoint su train, validation e test hold-out."
    )
    parser.add_argument("--model", default="best_model/best_model")
    parser.add_argument("--data-folder", default="data")
    parser.add_argument("--test-folder", default="data/test")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--operational-tolerance", type=float, default=0.20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-solution-details",
        action="store_true",
        help="Non stampare la distanza di ogni singola rotta (resta comunque nel JSON).",
    )
    parser.add_argument(
        "--output", default="recommender/final_comparison_results.json"
    )
    args = parser.parse_args()

    report = compare_train_validation_test(
        args.model,
        data_folder=args.data_folder,
        test_folder=args.test_folder,
        validation_fraction=args.validation_fraction,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        n_shuffles=args.shuffles,
        operational_tolerance=args.operational_tolerance,
        verbose=args.verbose,
        show_solution_distances=not args.no_solution_details,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report completo salvato in {output}")


if __name__ == "__main__":
    main()
