from __future__ import annotations

import math
import random
from typing import Sequence


FITNESSES = {
    "f1": {"index": 0, "name": "path_length", "unit": "NM"},
    "f2": {"index": 1, "name": "horizontal_target_error_nm", "unit": "NM"},
    "f3": {"index": 2, "name": "final_bearing_error_deg", "unit": "deg"},
}
DECISION_PROFILES = ("ppo", "best", "median", "worst")


def ranked_indices(fitnesses: Sequence[Sequence[float]], fitness_index: int) -> list[int]:
    values = [float(row[fitness_index]) for row in fitnesses]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Le fitness devono essere finite")
    return sorted(range(len(values)), key=lambda index: (values[index], index))


def alternative_indices(
    fitnesses: Sequence[Sequence[float]], ppo_index: int, fitness_index: int
) -> list[int]:
    if len(fitnesses) < 5:
        raise ValueError("Servono almeno 5 rotte per PPO + 4 alternative")
    if not 0 <= ppo_index < len(fitnesses):
        raise IndexError("Indice PPO fuori dal fronte")
    order = ranked_indices(fitnesses, fitness_index)
    last = len(order) - 1
    target_ranks = (0, last // 2, last, last // 4, (3 * last) // 4)
    selected = [ppo_index]
    for rank in target_ranks:
        candidate = order[rank]
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == 5:
            return selected
    for candidate in order:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == 5:
            return selected
    raise RuntimeError("Impossibile costruire cinque scelte uniche")


def decision_index(
    profile: str,
    fitnesses: Sequence[Sequence[float]],
    ppo_index: int,
    fitness_index: int,
) -> int:
    if profile == "ppo":
        return ppo_index
    order = ranked_indices(fitnesses, fitness_index)
    positions = {"best": 0, "median": (len(order) - 1) // 2, "worst": -1}
    if profile not in positions:
        raise ValueError(f"Profilo decisionale sconosciuto: {profile}")
    return order[positions[profile]]


def classify_decision(
    selected_index: int,
    fitnesses: Sequence[Sequence[float]],
    ppo_index: int,
    fitness_index: int,
) -> str:
    if selected_index == ppo_index:
        return "ppo"
    order = ranked_indices(fitnesses, fitness_index)
    labels = {order[0]: "best", order[(len(order) - 1) // 2]: "median", order[-1]: "worst"}
    return labels.get(selected_index, "other")


def parse_mix(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        try:
            name, raw_weight = item.strip().split("=", 1)
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError("--mix richiede valori come ppo=25,best=25,median=25,worst=25") from exc
        if name not in DECISION_PROFILES:
            raise ValueError(f"Profilo non valido in --mix: {name}")
        if name in weights or weight < 0 or not math.isfinite(weight):
            raise ValueError(f"Peso non valido in --mix: {item}")
        weights[name] = weight
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("--mix deve contenere almeno un peso positivo")
    return weights


def mixed_schedule(count: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    total = sum(weights.values())
    quotas = {name: count * weight / total for name, weight in weights.items()}
    allocated = {name: math.floor(quota) for name, quota in quotas.items()}
    missing = count - sum(allocated.values())
    for name in sorted(weights, key=lambda item: (-(quotas[item] - allocated[item]), item))[:missing]:
        allocated[name] += 1
    schedule = [name for name in DECISION_PROFILES for _ in range(allocated.get(name, 0))]
    rng.shuffle(schedule)
    return schedule
