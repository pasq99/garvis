"""Operatori NSGA-II e simulazione di spezzate 4D a quota costante.

Il cromosoma resta composto da geni ``(tempo_sec, delta_prua_deg,
velocita_kmh)``.  La quota non e' un gene: una soluzione non puo' quindi
salire (ne' scendere). Ogni gene cambia prua e velocita' all'istante indicato.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Iterable, List, Sequence, Tuple

from ga.fuzzy_coverage import direct_distance, fuzzy_cell
from scenario.conf import scn

Gene = Tuple[float, float, float]
Individuo = List[Gene]
KMH_TO_NM_SEC = 0.539957 / 3600.0
EPS = 1.0e-9


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def ordina_individuo(individuo: Individuo) -> Individuo:
    return sorted(individuo, key=lambda gene: gene[0])


def random_gene(time_span: Sequence[float], speed_lim: Sequence[float], max_dtheta: float = 90.0) -> Gene:
    return (
        random.uniform(float(time_span[0]), float(time_span[1])),
        random.uniform(-max_dtheta, max_dtheta),
        random.uniform(float(speed_lim[0]), float(speed_lim[1])),
    )


def init_individuo_random(num_geni: int, speed_lim: Sequence[float], theta_lim: float,
                          time_span: Sequence[float]) -> Individuo:
    """Crea esattamente ``num_geni``; tutti gli operatori mantengono tale cardinalita'."""
    return ordina_individuo([
        random_gene(time_span, speed_lim, theta_lim) for _ in range(num_geni)
    ])


def mutazione_gene(individuo: Individuo, sigma_t: float | None = None,
                   sigma_theta: float | None = None, sigma_v: float | None = None,
                   gene_probability: float | None = None,
                   reset_probability: float | None = None) -> Individuo:
    """Muta piu' geni e include un raro reset, evitando copie nella stessa nicchia."""
    nuovo = list(individuo)
    if not nuovo:
        return nuovo
    sigma_t = float(sigma_t if sigma_t is not None else scn.get("mutation_sigma_time_sec", 20.0))
    sigma_theta = float(sigma_theta if sigma_theta is not None else scn.get("mutation_sigma_heading_deg", 8.0))
    sigma_v = float(sigma_v if sigma_v is not None else scn.get("mutation_sigma_speed_kmh", 25.0))
    gene_probability = float(gene_probability if gene_probability is not None else scn.get("mutation_gene_probability", 0.35))
    reset_probability = float(reset_probability if reset_probability is not None else scn.get("mutation_reset_probability", 0.0))
    max_time = float(scn.get("MAX_TIME", scn["t_max_sim"]))
    mutated = False
    for index, (tempo, angolo, velocita) in enumerate(nuovo):
        if random.random() >= gene_probability:
            continue
        mutated = True
        if random.random() < reset_probability:
            nuovo[index] = random_gene((0.0, max_time),
                                       (scn["speed_min"], scn["speed_max"]),
                                       scn["max_turn_angle_deg"])
            continue
        # Mutare insieme i tre campi produce variazioni geometriche apprezzabili.
        tempo = clamp(tempo + random.gauss(0.0, sigma_t), 0.0, max_time)
        angolo = clamp(angolo + random.gauss(0.0, sigma_theta),
                       -scn["max_turn_angle_deg"], scn["max_turn_angle_deg"])
        velocita = clamp(velocita + random.gauss(0.0, sigma_v),
                         scn["speed_min"], scn["speed_max"])
        nuovo[index] = (tempo, angolo, velocita)
    if not mutated:
        index = random.randrange(len(nuovo))
        tempo, angolo, velocita = nuovo[index]
        nuovo[index] = (tempo, clamp(angolo + random.gauss(0.0, sigma_theta),
                                    -scn["max_turn_angle_deg"], scn["max_turn_angle_deg"]), velocita)
    return ordina_individuo(nuovo)


def crossover_un_punto(primo: Individuo, secondo: Individuo) -> tuple[Individuo, Individuo]:
    """Crossover posizionale: non aggiunge o elimina geni."""
    if len(primo) != len(secondo) or len(primo) < 2:
        return list(primo), list(secondo)
    cut = random.randrange(1, len(primo))
    return (ordina_individuo(list(primo[:cut]) + list(secondo[cut:])),
            ordina_individuo(list(secondo[:cut]) + list(primo[cut:])))


def _ray_exit(x: float, y: float, heading_deg: float, side: float) -> tuple[float, float, float]:
    dx, dy = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    candidates: list[float] = []
    if dx > EPS:
        candidates.append((side - x) / dx)
    elif dx < -EPS:
        candidates.append(-x / dx)
    if dy > EPS:
        candidates.append((side - y) / dy)
    elif dy < -EPS:
        candidates.append(-y / dy)
    positive = [value for value in candidates if value > EPS]
    if not positive:
        raise ValueError("Impossibile determinare l'intersezione con il bordo")
    distance = min(positive)
    return clamp(x + dx * distance, 0.0, side), clamp(y + dy * distance, 0.0, side), distance


def simula_traiettoria(individuo: Individuo, p0: Sequence[float], theta0_deg: float | None = None,
                       v0: float | None = None, t0: float = 0.0, lato: float | None = None,
                       p_target_scenario: Sequence[float] | None = None, **legacy: Any) -> dict[str, Any]:
    """Simula segmenti rettilinei ``[x, y, z_ft, t_sec]`` fino al bordo."""
    if theta0_deg is None:
        theta0_deg = legacy.pop("theta0", None)
    if theta0_deg is None:
        if p_target_scenario is None:
            raise TypeError("theta0_deg oppure p_target_scenario e' richiesto")
        theta0_deg = math.degrees(math.atan2(p_target_scenario[1] - p0[1], p_target_scenario[0] - p0[0]))
    if v0 is None:
        v0 = float(legacy.pop("speed0", scn["speed_min"]))
    side = float(lato if lato is not None else scn["area_size"])
    x, y = float(p0[0]), float(p0[1])
    altitude = float(p0[2]) if len(p0) > 2 else 0.0
    heading, speed, current_time = float(theta0_deg), float(v0), float(t0)
    path = [[x, y, altitude, current_time]]
    segments: list[dict[str, Any]] = []
    path_length = 0.0
    angle_change = 0.0

    for event_time, delta_heading, new_speed in ordina_individuo(individuo):
        event_time = max(current_time, float(event_time))
        travel = speed * KMH_TO_NM_SEC * (event_time - current_time)
        candidate_x = x + travel * math.cos(math.radians(heading))
        candidate_y = y + travel * math.sin(math.radians(heading))
        if not (0.0 <= candidate_x <= side and 0.0 <= candidate_y <= side):
            exit_x, exit_y, distance = _ray_exit(x, y, heading, side)
            exit_time = current_time + distance / (speed * KMH_TO_NM_SEC)
            segments.append({"p0": [x, y, altitude], "p1": [exit_x, exit_y, altitude],
                             "t0": current_time, "t1": exit_time, "speed_kmh": speed})
            path.append([exit_x, exit_y, altitude, exit_time])
            path_length += distance
            return _simulation_result(path, segments, path_length, angle_change, heading)
        if event_time > current_time + EPS:
            segments.append({"p0": [x, y, altitude], "p1": [candidate_x, candidate_y, altitude],
                             "t0": current_time, "t1": event_time, "speed_kmh": speed})
            path.append([candidate_x, candidate_y, altitude, event_time])
            path_length += travel
        x, y, current_time = candidate_x, candidate_y, event_time
        heading += float(delta_heading)
        speed = clamp(float(new_speed), scn["speed_min"], scn["speed_max"])
        angle_change += abs(float(delta_heading))

    exit_x, exit_y, distance = _ray_exit(x, y, heading, side)
    exit_time = current_time + distance / (speed * KMH_TO_NM_SEC)
    segments.append({"p0": [x, y, altitude], "p1": [exit_x, exit_y, altitude],
                     "t0": current_time, "t1": exit_time, "speed_kmh": speed})
    path.append([exit_x, exit_y, altitude, exit_time])
    path_length += distance
    return _simulation_result(path, segments, path_length, angle_change, heading)


def _simulation_result(path: list[list[float]], segments: list[dict[str, Any]], length: float,
                       angle_change: float, final_heading: float) -> dict[str, Any]:
    return {"path": path, "segments": segments, "exited": True,
            "time_of_exit": path[-1][3], "exit_point": path[-1][:3],
            "path_length_inside": length, "total_angle_change": angle_change,
            "final_heading_deg": final_heading % 360.0}


def nominal_simulation(aircraft: dict[str, Any]) -> dict[str, Any]:
    return simula_traiettoria([], aircraft["p0"], aircraft["prua0"], aircraft["v0"],
                             lato=scn["area_size"], p_target_scenario=aircraft["p_target"])


def minimum_4d_separation(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    """Minimo continuo sincronizzato tra due spezzate (non semplice campionamento)."""
    best_h = math.inf
    best_v = math.inf
    best_t = 0.0
    for a in first["segments"]:
        for b in second["segments"]:
            start, end = max(a["t0"], b["t0"]), min(a["t1"], b["t1"])
            if end < start - EPS:
                continue
            def state(segment: dict[str, Any], time_value: float) -> tuple[float, float, float]:
                duration = segment["t1"] - segment["t0"]
                ratio = 0.0 if duration <= EPS else (time_value - segment["t0"]) / duration
                return tuple(segment["p0"][i] + ratio * (segment["p1"][i] - segment["p0"][i]) for i in range(3))
            pa, pb = state(a, start), state(b, start)
            duration = end - start
            va = tuple((a["p1"][i] - a["p0"][i]) / (a["t1"] - a["t0"]) for i in range(2))
            vb = tuple((b["p1"][i] - b["p0"][i]) / (b["t1"] - b["t0"]) for i in range(2))
            rx, ry = pa[0] - pb[0], pa[1] - pb[1]
            vx, vy = va[0] - vb[0], va[1] - vb[1]
            denom = vx * vx + vy * vy
            offset = clamp(-(rx * vx + ry * vy) / denom, 0.0, duration) if denom > EPS else 0.0
            horizontal = math.hypot(rx + vx * offset, ry + vy * offset)
            vertical = abs(pa[2] - pb[2])
            if horizontal < best_h:
                best_h, best_v, best_t = horizontal, vertical, start + offset
    return {"horizontal_nm": best_h, "vertical_ft": best_v, "time_sec": best_t}


def _angle_error(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def valuta_individuo(individuo: Individuo, exp_scenario: dict[str, Any]):
    controlled, intruder = exp_scenario["ac0"], exp_scenario["ac1"]
    simulation = simula_traiettoria(individuo, controlled["p0"], controlled["prua0"], controlled["v0"],
                                    lato=scn["area_size"], p_target_scenario=controlled["p_target"])
    if any(abs(simulation["path"][0][axis] - float(controlled["p0"][axis])) > EPS for axis in range(3)):
        raise RuntimeError("La traiettoria AC0 non parte dal p0 dello scenario")
    reference = exp_scenario.get("intruder_sim") or nominal_simulation(intruder)
    separation = minimum_4d_separation(simulation, reference)
    conflict = (separation["horizontal_nm"] < scn["separation_min"] and
                separation["vertical_ft"] < scn["separation_min_vertical"])
    violation = max(0.0, scn["separation_min"] - separation["horizontal_nm"]) if conflict else 0.0
    target = controlled["p_target"]
    horizontal_error = math.hypot(simulation["exit_point"][0] - target[0], simulation["exit_point"][1] - target[1])
    target_tolerance = float(scn["target_error_tolerance_nm"])
    target_violation = max(0.0, horizontal_error - target_tolerance)
    heading_error = _angle_error(simulation["final_heading_deg"], controlled.get("prua_f", controlled["prua0"]))
    fitness = (
        simulation["path_length_inside"], horizontal_error, heading_error,
    )
    constraint_violation = violation + target_violation
    feasible = not conflict and target_violation <= EPS
    if not feasible:
        fitness = tuple(value + 1.0e6 + constraint_violation * 1.0e5 for value in fitness)
    simulation.update({"minimum_separation": separation, "conflict": conflict,
                       "target_reached": target_violation <= EPS,
                       "target_error_tolerance_nm": target_tolerance,
                       "target_constraint_violation_nm": target_violation,
                       "separation_constraint_violation_nm": violation,
                       "constraint_violation": constraint_violation,
                       "feasible": feasible,
                       "fuzzy_cell": list(fuzzy_cell(
                           simulation["path_length_inside"], horizontal_error,
                           heading_error, direct_distance(exp_scenario),
                       ))})
    return fitness, simulation


def save_pareto_front(
    exp_scenario: dict[str, Any],
    population: Iterable[Any],
    filename: str,
    selection: dict[str, Any] | None = None,
) -> str:
    entries = []
    for index, individual in enumerate(population):
        simulation = individual.sim
        raw_fitness = (simulation["path_length_inside"],
                       math.hypot(simulation["exit_point"][0] - exp_scenario["ac0"]["p_target"][0],
                                  simulation["exit_point"][1] - exp_scenario["ac0"]["p_target"][1]),
                       _angle_error(simulation["final_heading_deg"], exp_scenario["ac0"].get("prua_f", exp_scenario["ac0"]["prua0"])))
        entries.append({"solution_index": index, "genome": [list(gene) for gene in individual],
                        "pareto_rank": int(simulation.get("pareto_rank", 0)),
                        "fuzzy_cell": list(simulation.get("fuzzy_cell", [])),
                        "fitness": list(raw_fitness), "path_length": raw_fitness[0],
                        "horizontal_target_error_nm": raw_fitness[1], "final_bearing_error_deg": raw_fitness[2],
                        "path": simulation["path"], "segments": simulation["segments"],
                        "exit_point": simulation["exit_point"], "time_of_exit": simulation["time_of_exit"],
                        "minimum_separation": simulation["minimum_separation"],
                        "conflict": simulation["conflict"], "feasible": simulation["feasible"],
                        "target_reached": simulation["target_reached"],
                        "target_error_tolerance_nm": simulation["target_error_tolerance_nm"],
                        "target_constraint_violation_nm": simulation["target_constraint_violation_nm"],
                        "separation_constraint_violation_nm": simulation["separation_constraint_violation_nm"],
                        "constraint_violation": simulation["constraint_violation"]})
    document = {"scenario": {"area_size": scn["area_size"],
                              "separation_min": scn["separation_min"],
                              "separation_min_vertical": scn["separation_min_vertical"],
                              "target_error_tolerance_nm": scn["target_error_tolerance_nm"],
                              "ac0": exp_scenario["ac0"], "ac1": exp_scenario["ac1"]},
                "fixed_altitude_ft": exp_scenario["ac0"]["p0"][2],
                "selection": selection or {"mode": "exact_pareto"},
                "population": entries}
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
    return filename
