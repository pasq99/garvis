
from __future__ import annotations

from typing import Any

import numpy as np

from distillation.fuzzy import FuzzyBasis, MEMBERSHIPS, RULES, _membership


OUTPUT_TERMS = tuple(MEMBERSHIPS["output"])


def _output_membership(values: np.ndarray) -> np.ndarray:
    return np.column_stack([
        _membership(values, *MEMBERSHIPS["output"][term]) for term in OUTPUT_TERMS
    ])


def _output_centers() -> np.ndarray:
    universe = np.linspace(0.0, 100.0, 10_001)
    degrees = _output_membership(universe)
    return (universe[:, None] * degrees).sum(axis=0) / degrees.sum(axis=0)


def _to_fuzzy_scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("Calibrazione Wang-Mendel non valida")
    return np.clip((np.asarray(values, dtype=np.float64) - low) * 100.0 / (high - low), 0.0, 100.0)


def fit_wang_mendel(
    basis: FuzzyBasis,
    features: np.ndarray,
    targets: np.ndarray,
    low: float,
    high: float,
) -> dict[str, Any]:
    input_degree = basis.raw_activation(features)
    input_rule = np.argmax(input_degree, axis=1)
    output_degree = _output_membership(_to_fuzzy_scale(targets, low, high))
    output_term = np.argmax(output_degree, axis=1)
    rows = np.arange(len(features))
    candidate_degree = input_degree[rows, input_rule] * output_degree[rows, output_term]

    consequents = np.full(len(RULES), -1, dtype=int)
    rule_degrees = np.zeros(len(RULES), dtype=np.float64)
    support = np.bincount(input_rule, minlength=len(RULES))
    for row, rule in enumerate(input_rule):
        if candidate_degree[row] > rule_degrees[rule]:
            consequents[rule] = output_term[row]
            rule_degrees[rule] = candidate_degree[row]

    return {
        "method": "wang_mendel_max_degree_conflict_resolution",
        "parameter_count": int(np.count_nonzero(consequents >= 0)),
        "output_terms": list(OUTPUT_TERMS),
        "output_centers": _output_centers().tolist(),
        "rule_consequents": [OUTPUT_TERMS[value] if value >= 0 else None for value in consequents],
        "rule_degrees": rule_degrees.tolist(),
        "rule_support": support.astype(int).tolist(),
    }


def predict_wang_mendel(
    basis: FuzzyBasis,
    features: np.ndarray,
    model: dict[str, Any],
    low: float,
    high: float,
) -> np.ndarray:
    term_index = {term: index for index, term in enumerate(model["output_terms"])}
    consequents = np.asarray([
        term_index[term] if term is not None else -1 for term in model["rule_consequents"]
    ])
    present = consequents >= 0
    if not np.any(present):
        raise ValueError("Wang-Mendel non ha generato regole")
    activation = basis.raw_activation(features)[:, present]
    denominator = activation.sum(axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("Nessuna regola Wang-Mendel attiva per alcuni input")
    centers = np.asarray(model["output_centers"], dtype=np.float64)[consequents[present]]
    fuzzy_output = activation @ centers / denominator
    return low + fuzzy_output * (high - low) / 100.0


def self_check() -> None:
    basis = FuzzyBasis()
    features = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    model = fit_wang_mendel(basis, features, np.asarray([-1.0, 0.0, 1.0]), -1.0, 1.0)
    assert model["parameter_count"] == 3
    assert np.isfinite(predict_wang_mendel(basis, features, model, -1.0, 1.0)).all()


if __name__ == "__main__":
    self_check()
