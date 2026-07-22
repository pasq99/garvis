"""Report the highest-regret decisions on the exhaustive test set."""

from __future__ import annotations

try:
    from recommender.rlfuzzy import evaluate_model
except ImportError:
    from rlfuzzy import evaluate_model


def run_xai_analysis(folder_test: str, model_path: str, threshold: float = 20.0) -> None:
    summary = evaluate_model(folder_test, model_path, verbose=False)
    failures = sorted(
        (item for item in summary["detailed_results"] if item["fuzzy_deviation"] > threshold),
        key=lambda item: item["fuzzy_deviation"],
        reverse=True,
    )
    print(f"Decisioni con regret fuzzy > {threshold}: {len(failures)}")
    for item in failures:
        print(
            f"{item['filename']} | n={item['n_routes']} | "
            f"slot={item['suggested_idx']} (origine={item['suggested_source_idx']}) | "
            f"ottimo={item['chosen_idx']} (origine={item['chosen_source_idx']}) | "
            f"regret={item['fuzzy_deviation']:.2f}"
        )


if __name__ == "__main__":
    run_xai_analysis("./data/test", "./best_model/best_model")
