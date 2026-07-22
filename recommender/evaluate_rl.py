"""Compatibility entry point for bandit evaluation."""

try:
    from recommender.evaluate_fuzzy import main
except ImportError:
    from evaluate_fuzzy import main


if __name__ == "__main__":
    main()
