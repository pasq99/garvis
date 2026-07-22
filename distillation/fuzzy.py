"""The fixed 27-rule fuzzy basis and quadratic TSK design matrix."""

from __future__ import annotations

import hashlib
import json

import numpy as np


PATH_TERMS = ("direct", "moderate", "long")
HORIZONTAL_TERMS = ("precise", "acceptable", "limit")
BEARING_TERMS = ("aligned", "deviated", "poor")
RULES = tuple(
    (path, horizontal, bearing)
    for path in PATH_TERMS for horizontal in HORIZONTAL_TERMS for bearing in BEARING_TERMS
)
TSK_TERMS = (
    "bias", "path", "horizontal", "bearing", "path_squared",
    "horizontal_squared", "bearing_squared", "path_horizontal",
    "path_bearing", "horizontal_bearing",
)
MEMBERSHIPS = {
    "path": {
        "direct": ("trap", (1.0, 1.0, 1.03, 1.10)),
        "moderate": ("tri", (1.05, 1.15, 1.30)),
        "long": ("trap", (1.20, 1.40, 2.50, 2.50)),
    },
    "horizontal": {
        "precise": ("trap", (0.0, 0.0, 0.005, 0.020)),
        "acceptable": ("tri", (0.010, 0.040, 0.075)),
        "limit": ("trap", (0.055, 0.085, 0.10, 0.10)),
    },
    "bearing": {
        "aligned": ("trap", (0.0, 0.0, 1.0, 3.0)),
        "deviated": ("tri", (2.0, 10.0, 30.0)),
        "poor": ("trap", (20.0, 45.0, 180.0, 180.0)),
    },
    "output": {
        "very_low": ("trap", (0.0, 0.0, 10.0, 25.0)),
        "low": ("tri", (15.0, 30.0, 45.0)),
        "medium": ("tri", (35.0, 50.0, 65.0)),
        "high": ("tri", (55.0, 72.0, 85.0)),
        "optimal": ("trap", (80.0, 90.0, 100.0, 100.0)),
    },
}


def normalized_to_physical(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    return np.column_stack((1.0 + 1.5 * values[..., 0], 0.10 * values[..., 1], 180.0 * values[..., 2]))


def physical_to_normalized(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    return np.column_stack(((values[..., 0] - 1.0) / 1.5, values[..., 1] / 0.10, values[..., 2] / 180.0))


def _membership(values: np.ndarray, kind: str, points: tuple[float, ...]) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if kind == "tri":
        a, b, c = points
        result = np.where((a < values) & (values < b), (values - a) / (b - a), result)
        result = np.where(values == b, 1.0, result)
        return np.where((b < values) & (values < c), (c - values) / (c - b), result)
    a, b, c, d = points
    if b > a:
        result = np.where((a < values) & (values < b), (values - a) / (b - a), result)
    result = np.where((b <= values) & (values <= c), 1.0, result)
    if d > c:
        result = np.where((c < values) & (values < d), (d - values) / (d - c), result)
    return result


def basis_signature() -> str:
    return hashlib.sha256(json.dumps({"memberships": MEMBERSHIPS, "rules": RULES}, sort_keys=True).encode()).hexdigest()


class FuzzyBasis:
    def raw_activation(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(-1, 3)
        physical = normalized_to_physical(np.clip(values, 0.0, 1.0))
        groups = []
        for column, variable, terms in zip(
            physical.T,
            ("path", "horizontal", "bearing"),
            (PATH_TERMS, HORIZONTAL_TERMS, BEARING_TERMS),
        ):
            groups.append({term: _membership(column, *MEMBERSHIPS[variable][term]) for term in terms})
        return np.stack([
            groups[0][path] * groups[1][horizontal] * groups[2][bearing]
            for path, horizontal, bearing in RULES
        ], axis=1)

    def activation(self, features: np.ndarray) -> np.ndarray:
        activation = self.raw_activation(features)
        return activation / np.maximum(activation.sum(axis=1, keepdims=True), 1e-12)

    def design(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(-1, 3)
        x, y, z = values.T
        terms = np.column_stack((
            np.ones(len(values)), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z,
        ))
        return (self.activation(values)[:, :, None] * terms[:, None, :]).reshape(len(values), -1)

    def score(self, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return self.design(features) @ np.asarray(coefficients, dtype=np.float64).reshape(-1)

def self_check() -> None:
    basis = FuzzyBasis()
    assert len(RULES) == 27
    assert basis.design(np.zeros((2, 3))).shape == (2, len(RULES) * len(TSK_TERMS))


if __name__ == "__main__":
    self_check()
