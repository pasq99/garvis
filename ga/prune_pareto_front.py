"""Deduplicazione deterministica dei candidati 4D e split per scenario fisico."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
import random
import shutil
from typing import Any, Sequence

from ga.fuzzy_coverage import balanced_select, solution_fuzzy_cell
from scenario.conf import scn


DATA_DIR = "data"
PRUNED_DIRNAME = "pruned"
TEST_RATIO = 0.25
SEED = 42
TOL_H_NM = 0.1
TOL_PATH_NM = 1.0
TOL_HEADING_DEG = 2.0
TOL_GEOMETRY_NM = 0.5
ALTITUDE_TOLERANCE_FT = 1.0e-6
EPS = 1.0e-9


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def get_metric(solution: dict[str, Any], name: str, fitness_index: int | None = None) -> float | None:
    """Legge il contratto canonico, con fallback sui JSON precedenti."""

    value = solution.get(name)
    if value is None:
        value = (solution.get("metriche_reali") or {}).get(name)
    if value is None and fitness_index is not None:
        fitness = solution.get("fitness") or []
        if len(fitness) > fitness_index:
            value = fitness[fitness_index]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def canonical_metrics(solution: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    return (
        get_metric(solution, "path_length", fitness_index=0),
        get_metric(solution, "horizontal_target_error_nm", fitness_index=1),
        get_metric(solution, "final_bearing_error_deg", fitness_index=2),
    )


def path_descriptor(solution: dict[str, Any], samples: int = 13) -> list[tuple[float, float]] | None:
    path = solution.get("path") or []
    if len(path) < 2 or any(len(point) < 4 for point in path):
        return None
    cumulative = [0.0]
    try:
        for first, second in zip(path, path[1:]):
            cumulative.append(cumulative[-1] + math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            ))
    except (TypeError, ValueError):
        return None
    total = cumulative[-1]
    if total <= EPS:
        return None
    descriptor: list[tuple[float, float]] = []
    leg = 0
    for sample in range(samples):
        target = total * sample / max(samples - 1, 1)
        while leg + 1 < len(cumulative) and cumulative[leg + 1] < target:
            leg += 1
        end = min(leg + 1, len(path) - 1)
        length = cumulative[end] - cumulative[leg]
        ratio = (target - cumulative[leg]) / length if length > EPS else 0.0
        descriptor.append((
            float(path[leg][0]) + ratio * (float(path[end][0]) - float(path[leg][0])),
            float(path[leg][1]) + ratio * (float(path[end][1]) - float(path[leg][1])),
        ))
    return descriptor


def geometry_distance(first_solution: dict[str, Any], second_solution: dict[str, Any]) -> float:
    first = path_descriptor(first_solution)
    second = path_descriptor(second_solution)
    if first is None or second is None:
        return math.inf
    return math.sqrt(sum(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        for point1, point2 in zip(first, second)
    ) / max(len(first), 1))


def is_similar(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_cell, second_cell = first.get("fuzzy_cell"), second.get("fuzzy_cell")
    if first_cell and second_cell and first_cell != second_cell:
        return False
    metrics1, metrics2 = canonical_metrics(first), canonical_metrics(second)
    if any(value is None for value in (*metrics1, *metrics2)):
        return False
    path1, horizontal1, heading1 = metrics1
    path2, horizontal2, heading2 = metrics2
    assert path1 is not None and horizontal1 is not None and heading1 is not None
    assert path2 is not None and horizontal2 is not None and heading2 is not None
    return (
        abs(path1 - path2) <= TOL_PATH_NM
        and abs(horizontal1 - horizontal2) <= TOL_H_NM
        and abs(heading1 - heading2) <= TOL_HEADING_DEG
        and geometry_distance(first, second) <= TOL_GEOMETRY_NM
    )


def _candidate_validation_errors(
    solution: dict[str, Any], scenario_start: Sequence[float], scenario_target: Sequence[float],
) -> list[str]:
    errors: list[str] = []
    initial_altitude = float(scenario_start[2])
    if any(value is None for value in canonical_metrics(solution)):
        errors.append("metriche_geometriche_mancanti")
    genome = solution.get("genome") or []
    if not genome or any(not isinstance(gene, list) or len(gene) != 3 for gene in genome):
        errors.append("genoma_non_a_tre_campi")
    elif not all(_finite(value) for gene in genome for value in gene):
        errors.append("genoma_non_finito")
    path = solution.get("path") or []
    if len(path) < 2 or any(not isinstance(point, list) or len(point) != 4 for point in path):
        errors.append("path_non_4d")
    elif any(
        not all(_finite(value) for value in point)
        or abs(float(point[2]) - initial_altitude) > ALTITUDE_TOLERANCE_FT
        for point in path
    ):
        errors.append("path_non_finito_o_quota_variabile")
    elif any(abs(float(path[0][axis]) - float(scenario_start[axis])) > EPS for axis in range(3)):
        errors.append("partenza_diversa_da_ac0_p0")
    else:
        exit_error = math.hypot(
            float(path[-1][0]) - float(scenario_target[0]),
            float(path[-1][1]) - float(scenario_target[1]),
        )
        stored_error = get_metric(solution, "horizontal_target_error_nm", fitness_index=1)
        if stored_error is None or abs(exit_error - stored_error) > EPS:
            errors.append("exit_distance_incoerente_con_p_target")
        target_tolerance = float(scn["target_error_tolerance_nm"])
        if exit_error > target_tolerance + EPS:
            errors.append("target_non_raggiunto_entro_tolleranza")
    segments = solution.get("segments") or []
    if not segments:
        errors.append("segmenti_mancanti")
    previous_end: Sequence[float] | None = None
    previous_time: float | None = None
    for segment in segments:
        p0, p1 = segment.get("p0"), segment.get("p1")
        t0, t1 = segment.get("t0"), segment.get("t1")
        if (
            not p0 or not p1 or len(p0) != 3 or len(p1) != 3
            or not all(_finite(value) for value in (*p0, *p1, t0, t1))
        ):
            errors.append("segmento_non_4d")
            break
        if abs(float(p0[2]) - initial_altitude) > ALTITUDE_TOLERANCE_FT or abs(float(p1[2]) - initial_altitude) > ALTITUDE_TOLERANCE_FT:
            errors.append("quota_segmento_variabile")
            break
        if float(t1) <= float(t0):
            errors.append("tempo_segmento_non_monotono")
            break
        if previous_end is not None and (
            any(abs(float(first) - float(second)) > EPS for first, second in zip(previous_end, p0))
            or previous_time is None or abs(previous_time - float(t0)) > EPS
        ):
            errors.append("segmenti_discontinui")
            break
        previous_end, previous_time = p1, float(t1)
    real = solution.get("metriche_reali") or {}
    if not bool(solution.get("feasible", real.get("feasible", False))):
        errors.append("non_feasible")
    if bool(solution.get("conflict") or solution.get("collisione") or real.get("conflict")):
        errors.append("conflitto")
    violation = get_metric(solution, "constraint_violation")
    if violation is None or violation > EPS:
        errors.append("constraint_violation")
    return sorted(set(errors))


def quality_key(solution: dict[str, Any]) -> tuple[Any, ...]:
    path_length, horizontal_error, heading_error = canonical_metrics(solution)
    genome_key = tuple(float(value) for gene in solution.get("genome", []) for value in gene[:3])
    return (
        int(solution.get("pareto_rank", 10**9)),
        path_length if path_length is not None else math.inf,
        horizontal_error if horizontal_error is not None else math.inf,
        heading_error if heading_error is not None else math.inf,
        genome_key,
        int(solution.get("solution_index", 10**9)),
    )


def prune_file(file_path: str, output_dir: str) -> str | None:
    basename = os.path.basename(file_path)
    if "_pruned" in basename or "_infeasible" in basename:
        return None
    with open(file_path, encoding="utf-8") as handle:
        document = json.load(handle)
    population = document.get("population") or []
    scenario = document.get("scenario") or {}
    p0 = (scenario.get("ac0") or {}).get("p0") or []
    p_target = (scenario.get("ac0") or {}).get("p_target") or []
    if (
        not population or len(p0) != 3 or len(p_target) != 3
        or not all(_finite(value) for value in (*p0, *p_target))
    ):
        return None
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for solution in population:
        errors = _candidate_validation_errors(solution, p0, p_target)
        if errors:
            rejected.append({"solution_index": solution.get("solution_index"), "errors": errors})
        else:
            solution["fuzzy_cell"] = list(solution_fuzzy_cell(solution, document))
            valid.append(solution)
    filtered: list[dict[str, Any]] = []
    for solution in sorted(valid, key=quality_key):
        if not any(is_similar(solution, kept) for kept in filtered):
            filtered.append(solution)
    deduplicated_size = len(filtered)
    capacity = int(scn.get("pareto_archive_size", 100))
    if len(filtered) > capacity:
        filtered = balanced_select(
            filtered,
            capacity,
            cell_key=lambda item: tuple(item["fuzzy_cell"]),
            quality_key=quality_key,
        )
    for new_index, solution in enumerate(filtered):
        solution["source_solution_index"] = int(solution.get("solution_index", new_index))
        solution["solution_index"] = new_index
    document["population"] = filtered
    document["status"] = "ok" if filtered else "no_valid_candidates_after_pruning"
    document["pruning"] = {
        "source_population_size": len(population),
        "valid_source_candidates": len(valid),
        "rejected_invalid_candidates": len(rejected),
        "duplicates_removed": len(valid) - deduplicated_size,
        "capacity_removed": deduplicated_size - len(filtered),
        "pruned_population_size": len(filtered),
        "fuzzy_cell_counts": dict(sorted(Counter(
            "/".join(solution["fuzzy_cell"]) for solution in filtered
        ).items())),
        "stable_order": "pareto_rank_then_geometric_fitness_then_genome",
        "tolerances": {
            "path_length_nm": TOL_PATH_NM,
            "horizontal_target_error_nm": TOL_H_NM,
            "final_bearing_error_deg": TOL_HEADING_DEG,
            "geometry_rms_nm": TOL_GEOMETRY_NM,
        },
        "rejected": rejected,
    }
    os.makedirs(output_dir, exist_ok=True)
    root, extension = os.path.splitext(basename)
    output = os.path.join(output_dir, f"{root}_pruned{extension or '.json'}")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
    print(
        f"Pruned {basename}: {len(population)} -> {len(filtered)} "
        f"(invalidi={len(rejected)}, duplicati={len(valid) - deduplicated_size}, "
        f"oltre-capacita={deduplicated_size - len(filtered)})"
    )
    return output


def get_scenario_id(filename: str) -> str:
    return filename.split("_pareto")[0]


def split_by_scenario(pruned_files: Sequence[str], pruned_dir: str, test_ratio: float, seed: int) -> dict[str, Any]:
    train_dir = os.path.join(pruned_dir, "train")
    test_dir = os.path.join(pruned_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Include existing snapshots as well as newly pruned ones. This makes an
    # incremental pruning run atomic at physical-scenario level: generations
    # 50..250 can never be left across train and test.
    all_files = list(pruned_files)
    all_files.extend(glob.glob(os.path.join(train_dir, "*_pareto_front_*_pruned.json")))
    all_files.extend(glob.glob(os.path.join(test_dir, "*_pareto_front_*_pruned.json")))
    groups: dict[str, list[str]] = defaultdict(list)
    seen_paths: set[str] = set()
    for path in all_files:
        absolute = os.path.abspath(path)
        if absolute in seen_paths or not os.path.isfile(path):
            continue
        seen_paths.add(absolute)
        groups[get_scenario_id(os.path.basename(path))].append(path)

    # Lo scenario fisico e' l'unita' dello split: tutte le sue varianti restano
    # nello stesso insieme, evitando contaminazioni tra train e test.
    rng = random.Random(seed)
    identifiers = sorted(groups)
    test_count = 0 if len(identifiers) < 2 else min(
        len(identifiers) - 1,
        max(1, round(len(identifiers) * test_ratio)),
    )
    strata: dict[str, list[str]] = defaultdict(list)
    for scenario_id in identifiers:
        stratum = "legacy"
        try:
            with open(sorted(groups[scenario_id])[0], encoding="utf-8") as handle:
                document = json.load(handle)
            ac0 = ((document.get("scenario") or {}).get("ac0") or {})
            stratum = str(ac0.get("coverage_stratum", "legacy"))
        except (OSError, json.JSONDecodeError):
            pass
        strata[stratum].append(scenario_id)
    # Ranking proporzionale: con quattro scenari nello stesso strato, ad
    # esempio, il primo cade nel 25% test senza spezzare lo scenario fisico.
    ranked: list[tuple[float, str]] = []
    for stratum, members in sorted(strata.items()):
        rng.shuffle(members)
        ranked.extend(
            ((position + 0.5) / len(members), scenario_id)
            for position, scenario_id in enumerate(members)
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    test_ids = {scenario_id for _, scenario_id in ranked[:test_count]}
    moved = 0
    for scenario_id, files in sorted(groups.items()):
        destination_dir = test_dir if scenario_id in test_ids else train_dir
        other_dir = train_dir if scenario_id in test_ids else test_dir
        for source in sorted(files):
            if not os.path.exists(source):
                continue
            basename = os.path.basename(source)
            destination = os.path.join(destination_dir, basename)
            old_counterpart = os.path.join(other_dir, basename)
            if os.path.abspath(source) != os.path.abspath(destination):
                if os.path.exists(destination):
                    os.remove(destination)
                os.replace(source, destination)
                moved += 1
            if os.path.exists(old_counterpart):
                os.remove(old_counterpart)
    return {
        "physical_scenarios": len(groups),
        "train_scenarios": len(groups) - len(test_ids),
        "test_scenarios": len(test_ids),
        "files_placed": moved,
        "coverage_strata": len(strata),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default=DATA_DIR)
    parser.add_argument(
        "--output-subdir", default=PRUNED_DIRNAME,
        help="Sottodirectory di data_dir in cui salvare il pruning (default: pruned)",
    )
    parser.add_argument("--prune-only", action="store_true", help="Non crea lo split train/test")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not 0.0 < args.test_ratio < 1.0:
        parser.error("--test-ratio deve essere in (0, 1)")
    output_subdir = os.path.normpath(args.output_subdir)
    if (
        os.path.isabs(args.output_subdir)
        or output_subdir in ("", ".", os.pardir)
        or output_subdir.startswith(f"{os.pardir}{os.sep}")
    ):
        parser.error("--output-subdir deve essere una sottodirectory relativa di data_dir")
    originals = sorted(
        path for path in glob.glob(os.path.join(args.data_dir, "*_pareto_front_*.json"))
        if "_pruned" not in path and "_infeasible" not in path
    )
    pruned_dir = os.path.join(args.data_dir, output_subdir)
    staging_dir = os.path.join(pruned_dir, ".staging")
    os.makedirs(os.path.join(pruned_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(pruned_dir, "test"), exist_ok=True)
    outputs = [output for path in originals if (output := prune_file(path, staging_dir))]
    print(f"Creati {len(outputs)} file pruned da {len(originals)} originali.")
    output_names = {os.path.basename(path) for path in outputs}
    for original in originals:
        root, extension = os.path.splitext(os.path.basename(original))
        pruned_name = f"{root}_pruned{extension or '.json'}"
        if pruned_name in output_names:
            continue
        for split_dir in ("train", "test"):
            stale = os.path.join(pruned_dir, split_dir, pruned_name)
            if os.path.exists(stale):
                os.remove(stale)
    if outputs:
        if args.prune_only:
            train_dir = os.path.join(pruned_dir, "train")
            os.makedirs(train_dir, exist_ok=True)
            for source in outputs:
                shutil.move(source, os.path.join(train_dir, os.path.basename(source)))
        else:
            summary = split_by_scenario(outputs, pruned_dir, args.test_ratio, args.seed)
            print(f"Split: {summary}")
        try:
            os.rmdir(staging_dir)
        except OSError:
            pass
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
