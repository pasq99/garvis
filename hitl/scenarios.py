
from __future__ import annotations

import glob
import json
import math
import os
import re
from typing import Any

import numpy as np

from recommender.policy_contract import PADDING_VALUE, ROUTE_FEATURES


def is_conflicting(solution: dict[str, Any]) -> bool:
    real = solution.get("metriche_reali") or {}
    return bool(
        solution.get("collisione")
        or solution.get("conflict")
        or real.get("collisione")
        or real.get("conflict")
    )


def _metrics(solution: dict[str, Any]) -> tuple[float, float, float]:
    real = solution.get("metriche_reali") or {}
    fitness = solution.get("fitness") or []
    values = (
        solution.get("path_length", real.get("path_length", fitness[0] if len(fitness) > 0 else None)),
        solution.get(
            "horizontal_target_error_nm",
            real.get("horizontal_target_error_nm", fitness[1] if len(fitness) > 1 else None),
        ),
        solution.get(
            "final_bearing_error_deg",
            real.get("final_bearing_error_deg", fitness[2] if len(fitness) > 2 else None),
        ),
    )
    if any(value is None for value in values):
        raise ValueError("Fitness incomplete")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Fitness non finite")
    return result  # type: ignore[return-value]


def _direct_distance(document: dict[str, Any]) -> float:
    ac0 = (document.get("scenario") or {}).get("ac0") or {}
    start, target = ac0.get("p0"), ac0.get("p_target")
    if not start or not target or len(start) < 2 or len(target) < 2:
        raise ValueError("Punti iniziale/target mancanti")
    distance = math.hypot(float(target[0]) - float(start[0]), float(target[1]) - float(start[1]))
    if distance <= 0:
        raise ValueError("Distanza diretta non positiva")
    return distance


def _features(metrics: tuple[float, float, float], direct_distance: float) -> np.ndarray:
    path, horizontal, bearing = metrics
    if any(value >= 10_000.0 for value in metrics):
        return np.ones(len(ROUTE_FEATURES), dtype=np.float32)
    ratio = np.clip(path / direct_distance, 1.0, 2.5)
    return np.asarray([
        (ratio - 1.0) / 1.5,
        np.clip(horizontal, 0.0, 0.10) / 0.10,
        np.clip(abs(bearing), 0.0, 180.0) / 180.0,
    ], dtype=np.float32)


def load_scenarios(folder: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(folder, "*_pareto_front_*_pruned.json")))
    if not paths:
        raise ValueError(f"Nessun fronte pruned trovato in {folder}")
    scenarios = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            direct_distance = _direct_distance(document)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        routes, features = [], []
        for source_idx, solution in enumerate(document.get("population") or []):
            if is_conflicting(solution):
                continue
            try:
                values = _metrics(solution)
            except (TypeError, ValueError):
                continue
            routes.append({
                "source_idx": source_idx,
                "path_length": values[0],
                "horizontal_target_error_nm": values[1],
                "final_bearing_error_deg": values[2],
            })
            features.append(_features(values, direct_distance))
        if routes:
            filename = os.path.basename(path)
            match = re.match(r"((?:exp_)?\d+)_pareto_front_", filename)
            scenarios.append({
                "filename": filename,
                "group_id": match.group(1) if match else filename,
                "pareto_front": routes,
                "policy_features": np.asarray(features, dtype=np.float32),
            })
    if not scenarios:
        raise ValueError(f"Nessun fronte valido trovato in {folder}")
    print(f"Caricati {len(scenarios)} fronti HITL senza eseguire il fuzzy.")
    return scenarios


def model_input(scenario: dict[str, Any], capacity: int) -> tuple[np.ndarray, np.ndarray]:
    features = scenario["policy_features"]
    if len(features) > capacity:
        raise ValueError(f"Fronte da {len(features)} rotte > capacita' modello {capacity}")
    observation = np.full((capacity, len(ROUTE_FEATURES)), PADDING_VALUE, dtype=np.float32)
    observation[:len(features)] = features
    mask = np.zeros(capacity, dtype=bool)
    mask[:len(features)] = True
    return observation.reshape(-1), mask
