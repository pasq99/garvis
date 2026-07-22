
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from distillation.data import (
    Front, assert_disjoint, load_fronts, load_policy, ppo_logits, sha256_file,
    split_train_validation,
)
from distillation.fuzzy import FuzzyBasis, RULES, TSK_TERMS, basis_signature
from distillation.wang_mendel import fit_wang_mendel, predict_wang_mendel
from recommender.policy_contract import ROUTE_FEATURES


RIDGES = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
SERIOUS_REGRET = 0.9


def _flatten(fronts: Sequence[Front], logits: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([front.features for front in fronts]).astype(np.float64),
        np.concatenate(logits).astype(np.float64),
        np.asarray([len(front.features) for front in fronts], dtype=np.int32),
    )


def metrics(sizes: np.ndarray, teacher: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    top1 = top2 = mismatches = serious = pairs = matching_pairs = 0
    regrets = []
    offset = 0
    for size in sizes:
        reference = teacher[offset:offset + size]
        scores = predicted[offset:offset + size]
        expected = int(np.argmax(reference))
        order = np.argsort(-scores, kind="stable")
        selected = int(order[0])
        top1 += selected == expected
        top2 += expected in order[:min(2, size)]
        mismatches += selected != expected
        probability = np.exp(reference - reference.max())
        probability /= probability.sum()
        regret = float(probability[expected] - probability[selected])
        regrets.append(regret)
        serious += regret > SERIOUS_REGRET
        for first in range(size):
            for second in range(first + 1, size):
                if np.isclose(reference[first], reference[second]):
                    continue
                pairs += 1
                matching_pairs += np.sign(reference[first] - reference[second]) == np.sign(scores[first] - scores[second])
        offset += size
    error = predicted - teacher
    return {
        "fronts": int(len(sizes)),
        "routes": int(sizes.sum()),
        "top1_fidelity": top1 / len(sizes),
        "top2_fidelity": top2 / len(sizes),
        "pairwise_agreement": matching_pairs / pairs if pairs else 1.0,
        "mismatches": int(mismatches),
        "mean_regret": float(np.mean(regrets)),
        "max_regret": float(np.max(regrets)),
        "serious_regret_threshold": SERIOUS_REGRET,
        "serious_errors": int(serious),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def _selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    return (
        report["top1_fidelity"], report["pairwise_agreement"],
        -report["mean_regret"], -report["rmse"],
    )


def _compressed_qr_system(
    basis: FuzzyBasis, features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = len(RULES) * len(TSK_TERMS)
    sum_squares = np.zeros(width)
    for start in range(0, len(features), 8192):
        design = basis.design(features[start:start + 8192])
        sum_squares += np.einsum("ij,ij->j", design, design)
    scales = np.sqrt(sum_squares / len(features))
    scales[scales < np.finfo(np.float64).eps] = 1.0
    roots, projected = [], []
    normalizer = np.sqrt(len(features))
    # ponytail: QR per blocchi evita sia X completa sia le instabili equazioni normali.
    for start in range(0, len(features), 8192):
        design = basis.design(features[start:start + 8192]) / scales / normalizer
        target = targets[start:start + 8192] / normalizer
        orthogonal, root = np.linalg.qr(design, mode="reduced")
        roots.append(root)
        projected.append(orthogonal.T @ target)
    return np.vstack(roots), np.concatenate(projected), scales


def _solve_ridge(
    compressed: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    width = compressed.shape[1]
    augmented = np.vstack((compressed, np.sqrt(ridge) * np.eye(width)))
    right = np.concatenate((targets, np.zeros(width)))
    standardized, _, rank, singular = np.linalg.lstsq(augmented, right, rcond=None)
    coefficients = standardized / scales
    diagnostics = {
        "solver": "blocked_qr_numpy_lstsq",
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "standardized_coefficient_max_abs": float(np.abs(standardized).max()),
        "standardized_coefficient_l2": float(np.linalg.norm(standardized)),
        "coefficient_max_abs": float(np.abs(coefficients).max()),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
    }
    return coefficients, standardized, diagnostics


def _predict(basis: FuzzyBasis, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.concatenate([
        basis.score(features[start:start + 8192], coefficients)
        for start in range(0, len(features), 8192)
    ])


def calibration_from_artifact(artifact: dict[str, Any]) -> tuple[float, float]:
    calibration = artifact["calibration"]
    return float(calibration["low_logit"]), float(calibration["high_logit"])


def fit(
    model_path: str | Path,
    train_folder: str | Path,
    test_folder: str | Path,
    output: str | Path,
    *,
    ridge: float | None = None,
    calibration_from: str | Path | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    if ridge is not None and ridge <= 0.0:
        raise ValueError("ridge deve essere positivo")
    model, checkpoint, contract = load_policy(model_path, device)
    all_train, test_fronts = load_fronts(train_folder), load_fronts(test_folder)
    assert_disjoint(all_train, test_fronts)
    train_fronts, validation_fronts, validation_groups = split_train_validation(all_train, contract, seed)
    train_logits = ppo_logits(model, train_fronts)
    validation_logits = ppo_logits(model, validation_fronts)
    test_logits = ppo_logits(model, test_fronts)
    train_x, train_z, train_sizes = _flatten(train_fronts, train_logits)
    validation_x, validation_z, validation_sizes = _flatten(validation_fronts, validation_logits)
    test_x, test_z, test_sizes = _flatten(test_fronts, test_logits)
    if calibration_from:
        baseline = json.loads(Path(calibration_from).read_text(encoding="utf-8"))
        if baseline.get("basis_signature") != basis_signature():
            raise ValueError("Calibrazione baseline con base fuzzy differente")
        calibration = calibration_from_artifact(baseline)
        calibration_source = str(calibration_from)
    else:
        calibration = tuple(map(float, np.percentile(train_z, (1.0, 99.0))))
        if calibration[1] <= calibration[0]:
            raise ValueError("Logit PPO costanti: calibrazione impossibile")
        calibration_source = "training_logits_p01_p99"
    basis = FuzzyBasis()
    wang_mendel = fit_wang_mendel(basis, train_x, train_z, *calibration)
    wang_mendel.update({
        "train": metrics(
            train_sizes, train_z, predict_wang_mendel(basis, train_x, wang_mendel, *calibration)
        ),
        "validation": metrics(
            validation_sizes, validation_z,
            predict_wang_mendel(basis, validation_x, wang_mendel, *calibration),
        ),
        "test": metrics(
            test_sizes, test_z, predict_wang_mendel(basis, test_x, wang_mendel, *calibration)
        ),
    })
    candidates = (float(ridge),) if ridge is not None else RIDGES
    compressed, right, scales = _compressed_qr_system(basis, train_x, train_z)
    validation_design = basis.design(validation_x)
    trials = []
    coefficients_by_ridge = {}
    for value in candidates:
        coefficients, standardized, diagnostics = _solve_ridge(compressed, right, scales, value)
        coefficients_by_ridge[value] = (coefficients, standardized, diagnostics)
        trials.append({
            "ridge": value,
            "metrics": metrics(validation_sizes, validation_z, validation_design @ coefficients),
            "solver": diagnostics,
        })
    best_trial = max(trials, key=lambda item: _selection_key(item["metrics"]))
    best_ridge = float(best_trial["ridge"])
    coefficients, standardized, diagnostics = coefficients_by_ridge[best_ridge]
    quadratic = {
        "terms": list(TSK_TERMS),
        "parameter_count": len(RULES) * len(TSK_TERMS),
        "ridge": best_ridge,
        "coefficients": coefficients.reshape(len(RULES), len(TSK_TERMS)).tolist(),
        "standardized_coefficients": standardized.reshape(len(RULES), len(TSK_TERMS)).tolist(),
        "standardization": {
            "method": "train_column_rms",
            "column_scales": scales.reshape(len(RULES), len(TSK_TERMS)).tolist(),
        },
        "solver": diagnostics,
        "ridge_trials": trials,
        "train": metrics(train_sizes, train_z, _predict(basis, train_x, coefficients)),
        "validation": best_trial["metrics"],
        "test": metrics(test_sizes, test_z, _predict(basis, test_x, coefficients)),
    }
    artifact = {
        "format_version": 5,
        "type": "fuzzy_surrogate",
        "fit_methods": {
            "quadratic": "column_rms_standardization_and_blocked_qr_lstsq",
            "wang_mendel": "max_membership_and_max_degree_conflict_resolution",
        },
        "basis_signature": basis_signature(),
        "features": list(ROUTE_FEATURES),
        "model_sha256": sha256_file(checkpoint),
        "model_contract": contract,
        "validation_groups": validation_groups,
        "selection": ["top1_fidelity", "pairwise_agreement", "mean_regret", "rmse"],
        "selected_model": "quadratic",
        "calibration": {
            "low_logit": calibration[0], "high_logit": calibration[1],
            "source": calibration_source,
        },
        "models": {"quadratic": quadratic, "wang_mendel": wang_mendel},
        "selected_test": quadratic["test"],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact
