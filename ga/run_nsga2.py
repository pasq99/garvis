"""Esegue NSGA-II sulle spezzate 4D dello scenario corrente."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any, Sequence

from deap import base, creator, tools

from ga import deap_ga_module_v2 as ga
from ga.fuzzy_coverage import balanced_select
from scenario.conf import scn

logger = logging.getLogger(__name__)


def _path_descriptor(individual: Any, samples: int) -> list[tuple[float, float]]:
    path = individual.sim["path"]
    cumulative = [0.0]
    for first, second in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + ((second[0] - first[0]) ** 2 +
                                             (second[1] - first[1]) ** 2) ** 0.5)
    total = cumulative[-1]
    result = []
    leg = 0
    for sample in range(samples):
        target = total * sample / max(samples - 1, 1)
        while leg + 1 < len(cumulative) and cumulative[leg + 1] < target:
            leg += 1
        end = min(leg + 1, len(path) - 1)
        length = cumulative[end] - cumulative[leg]
        ratio = (target - cumulative[leg]) / length if length > ga.EPS else 0.0
        result.append((path[leg][0] + ratio * (path[end][0] - path[leg][0]),
                       path[leg][1] + ratio * (path[end][1] - path[leg][1])))
    return result


def _diverse_select(candidates: list[Any], size: int) -> list[Any]:
    """Applica NSGA-II, poi elimina copie nello spazio geometrico dei percorsi."""
    ordered = tools.selNSGA2(candidates, len(candidates))
    threshold = float(scn.get("diversity_geometry_nm", 0.0))
    samples = int(scn.get("diversity_samples", 13))
    if threshold <= 0.0:
        return ordered[:size]
    # Un rappresentante per nicchia impedisce che la pressione Pareto perda
    # completamente gli insiemi fuzzy rari durante l'evoluzione.
    niche_representatives: list[Any] = []
    seen_cells: set[tuple[str, ...]] = set()
    if bool(scn.get("pareto_fuzzy_coverage", True)):
        for individual in ordered:
            cell = tuple(individual.sim.get("fuzzy_cell", ()))
            if len(cell) == 3 and cell not in seen_cells:
                niche_representatives.append(individual)
                seen_cells.add(cell)
    kept, descriptors = [], []
    priority = niche_representatives + [
        item for item in ordered if id(item) not in {id(value) for value in niche_representatives}
    ]
    for individual in priority:
        descriptor = _path_descriptor(individual, samples)
        if all((sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                    for a, b in zip(descriptor, other)) / samples) ** 0.5 >= threshold
               for other in descriptors):
            kept.append(individual)
            descriptors.append(descriptor)
            if len(kept) == size:
                break
    # Se le nicchie disponibili sono meno di size, NSGA-II completa senza bloccare il run.
    if len(kept) < size:
        selected_ids = {id(item) for item in kept}
        kept.extend(item for item in ordered if id(item) not in selected_ids)
    return tools.selNSGA2(kept[:size], min(size, len(kept)))


def _scenario_contract(aircraft: dict[str, Any]) -> dict[str, Any]:
    return {
        "id_aereo": aircraft["id_aereo"], "p0": list(aircraft["p0"]),
        "p_target": list(aircraft["p_target"]), "prua0": aircraft["prua0"],
        "prua_f": aircraft["prua_f"], "v0": aircraft["v0"],
        "exit_time_sec": aircraft["exit_time_sec"], "tipo": aircraft["tipo"],
        "coverage_stratum": aircraft.get("coverage_stratum", "legacy"),
        "coverage_cell": aircraft.get("coverage_cell", "legacy"),
    }


def load_scenarios(path: str) -> dict[int, dict[str, Any]]:
    from scenario.scenario2d import leggi_scenari_csv

    loaded = leggi_scenari_csv(path)
    result: dict[int, dict[str, Any]] = {}
    for experiment, aircraft in loaded.items():
        by_id = {item["id_aereo"]: item for item in aircraft}
        if 0 in by_id and 1 in by_id:
            result[experiment] = {"ac0": _scenario_contract(by_id[0]), "ac1": _scenario_contract(by_id[1])}
    return result


def _ensure_deap_types() -> None:
    if not hasattr(creator, "FitnessGarvis3Objectives"):
        creator.create("FitnessGarvis3Objectives", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "IndividualGarvis3Objectives"):
        creator.create("IndividualGarvis3Objectives", list, fitness=creator.FitnessGarvis3Objectives, sim=dict)


def _genome_key(individual: Any) -> tuple[float, ...]:
    return tuple(round(float(value), 9) for gene in individual for value in gene)


def _archive_quality(individual: Any) -> tuple[Any, ...]:
    return (int(individual.sim.get("pareto_rank", 10**9)), *individual.fitness.values,
            _genome_key(individual))


def _update_archive(candidates: list[Any], capacity: int) -> tuple[list[Any], int]:
    """Keep exact Pareto plus fuzzy-balanced feasible candidates up to capacity."""
    feasible = [
        item for item in candidates
        if item.sim["feasible"] and item.sim["target_reached"]
        and item.sim["target_constraint_violation_nm"] <= ga.EPS
    ]
    unique = { _genome_key(item): item for item in feasible }
    pool = list(unique.values())
    if not pool:
        return [], 0
    fronts = tools.sortNondominated(pool, len(pool), first_front_only=False)
    for rank, front in enumerate(fronts):
        for item in front:
            item.sim["pareto_rank"] = rank
    exact = list(fronts[0])
    if len(exact) >= capacity:
        selected = balanced_select(
            exact, capacity,
            cell_key=lambda item: tuple(item.sim["fuzzy_cell"]),
            quality_key=_archive_quality,
        )
    else:
        remainder = [item for front in fronts[1:] for item in front]
        selected = balanced_select(
            remainder, capacity,
            initial=exact,
            cell_key=lambda item: tuple(item.sim["fuzzy_cell"]),
            quality_key=_archive_quality,
        )
    ordered = tools.selNSGA2(selected, len(selected))
    return ordered, sum(int(item.sim.get("pareto_rank", 0) == 0) for item in ordered)


def _save_generation_snapshot(
    experiment: int,
    generation: int,
    scenario: dict[str, Any],
    population: list[Any],
    output_dir: str,
    exact_pareto_size: int,
) -> tuple[str, int]:
    """Save the cumulative feasible archive and retain every candidate rank."""
    target_tolerance = float(scn["target_error_tolerance_nm"])
    feasible = [
        item for item in population
        if item.sim["feasible"]
        and item.sim["target_reached"]
        and item.sim["target_constraint_violation_nm"] <= ga.EPS
        and item.fitness.values[1] <= target_tolerance + ga.EPS
    ]
    output = os.path.join(output_dir, f"{experiment}_pareto_front_{generation}.json")
    selected = feasible[:int(scn.get("pareto_archive_size", 100))]
    ga.save_pareto_front(
        scenario, selected, output,
        selection={
            "mode": "exact_pareto_plus_fuzzy_coverage",
            "capacity": int(scn.get("pareto_archive_size", 100)),
            "exact_pareto_size": int(exact_pareto_size),
            "feasible_coverage_size": max(0, len(selected) - exact_pareto_size),
            "objectives": [
                "min_path_length", "min_horizontal_target_error",
                "min_final_bearing_error",
            ],
        },
    )
    logger.info(
        "Scenario %s, generazione %s: salvate %s soluzioni Pareto ammissibili",
        experiment, generation, len(selected),
    )
    return output, len(selected)


def run_experiment(experiment: int, scenario: dict[str, Any], output_dir: str,
                   pop_size: int, generations: int, seed: int | None = None,
                   snapshot_generations: Sequence[int] | None = None) -> str:
    if seed is not None:
        random.seed(seed + experiment)
    _ensure_deap_types()
    os.makedirs(output_dir, exist_ok=True)
    max_time = min(float(scenario["ac0"]["exit_time_sec"]), float(scn["t_max_sim"]))
    scn["MAX_TIME"] = max_time
    scenario["intruder_sim"] = ga.nominal_simulation(scenario["ac1"])

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.IndividualGarvis3Objectives(
        ga.init_individuo_random(scn["n_segments"], (scn["speed_min"], scn["speed_max"]),
                                 scn["max_turn_angle_deg"], (0.0, max_time))))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", ga.valuta_individuo, exp_scenario=scenario)
    toolbox.register("mate", ga.crossover_un_punto)
    toolbox.register("mutate", ga.mutazione_gene)
    toolbox.register("select", tools.selNSGA2)
    population = toolbox.population(n=pop_size)
    for individual in population:
        individual.fitness.values, individual.sim = toolbox.evaluate(individual)
    population = toolbox.select(population, len(population))

    configured_snapshots = (
        snapshot_generations
        if snapshot_generations is not None
        else scn.get("pareto_snapshot_generations", [50, 100, 150, 200, 250])
    )
    snapshots = {int(value) for value in configured_snapshots if 0 <= int(value) <= generations}
    snapshots.add(generations)
    saved_outputs: dict[int, str] = {}
    archive: list[Any] = []
    archive_capacity = int(scn.get("pareto_archive_size", 100))

    fitness_rows = []
    for generation in range(generations + 1):
        archive, exact_pareto_size = _update_archive(archive + population, archive_capacity)
        for individual in population:
            fitness_rows.append((experiment, generation, *individual.fitness.values,
                                 individual.sim["minimum_separation"]["horizontal_nm"],
                                 int(individual.sim["feasible"])))
        if generation in snapshots:
            output, _ = _save_generation_snapshot(
                experiment, generation, scenario, archive, output_dir, exact_pareto_size
            )
            saved_outputs[generation] = output
        if generation == generations:
            break
        # selTournamentDCD richiede k divisibile per quattro quando k == len(pop).
        offspring_count = pop_size - pop_size % 4
        parents = tools.selTournamentDCD(population, offspring_count)
        offspring = [toolbox.clone(item) for item in parents]
        for first, second in zip(offspring[::2], offspring[1::2]):
            if random.random() < float(scn.get("crossover_probability", 0.7)):
                first[:], second[:] = toolbox.mate(first, second)
            if random.random() < float(scn.get("mutation_probability", 0.3)):
                first[:] = toolbox.mutate(first)
            if random.random() < float(scn.get("mutation_probability", 0.3)):
                second[:] = toolbox.mutate(second)
            for item in (first, second):
                item.fitness.values, item.sim = toolbox.evaluate(item)
        immigrant_count = min(pop_size, round(pop_size * float(scn.get("random_immigrant_fraction", 0.0))))
        immigrants = toolbox.population(n=immigrant_count)
        for individual in immigrants:
            individual.fitness.values, individual.sim = toolbox.evaluate(individual)
        population = _diverse_select(population + offspring + immigrants, pop_size)

    log_path = os.path.join(output_dir, f"fitness_log_exp{experiment}_pop{pop_size}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "generation", "path_length",
                         "horizontal_target_error", "final_bearing_error",
                         "minimum_horizontal_separation_nm", "feasible"))
        writer.writerows(fitness_rows)
    logger.info("Scenario %s: snapshot salvati alle generazioni %s", experiment, sorted(saved_outputs))
    return saved_outputs[generations]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=scn["input_scenarios_file"])
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--first", type=int, default=scn["first_experiment"])
    parser.add_argument("--count", type=int, default=scn["n_experiments"])
    parser.add_argument("--population", type=int, default=scn["pop_size"])
    parser.add_argument("--generations", type=int, default=scn["n_generations"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=scn.get("n_worker", 16))
    parser.add_argument(
        "--snapshots", type=int, nargs="+",
        default=scn.get("pareto_snapshot_generations", [50, 100, 150, 200, 250]),
        help="Generazioni di cui salvare il fronte Pareto (la finale viene sempre inclusa).",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers deve essere positivo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    scenarios = load_scenarios(args.input)
    identifiers = [identifier for identifier in sorted(scenarios) if identifier >= args.first][:args.count]
    if not identifiers:
        parser.error("nessuno scenario completo nell'intervallo richiesto")
    start = time.time()
    logger.info("Avvio %s scenari con %s worker", len(identifiers), args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(
            run_experiment,
            identifiers,
            (scenarios[identifier] for identifier in identifiers),
            repeat(args.output_dir),
            repeat(args.population),
            repeat(args.generations),
            repeat(args.seed),
            repeat(args.snapshots),
            chunksize=1,
        ))
    logger.info("Completato in %.2f s", time.time() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
