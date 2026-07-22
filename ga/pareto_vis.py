"""Visualizza i candidati Pareto prodotti dal GA ATM a spezzate 4D.

Compatibile con:
- output corrente ``scenario`` + ``population``;
- metriche annidate in ``metriche_reali``;
- simulazione annidata in ``sim``;
- vecchi JSON 2D/tempo, quando presenti.

Per impostazione predefinita salva due PNG per ogni JSON. Usare ``--show`` per
aprire anche le figure quando è disponibile un backend grafico.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# In WSL/headless evita errori Tk/Qt; su Windows/macOS conserva il backend GUI.
if not os.environ.get("DISPLAY") and sys.platform not in {"win32", "darwin"}:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

EPS = 1.0e-9


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_point(value: Any, minimum_size: int = 2) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < minimum_size:
        return None
    converted: list[float] = []
    for item in value:
        number = _as_float(item)
        if number is None:
            return None
        converted.append(number)
    return converted


def _nested(solution: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Legge prima il campo canonico, poi metriche_reali e sim."""
    value = solution.get(key)
    if value is not None:
        return value
    metrics = solution.get("metriche_reali")
    if isinstance(metrics, Mapping) and metrics.get(key) is not None:
        return metrics.get(key)
    simulation = solution.get("sim")
    if isinstance(simulation, Mapping) and simulation.get(key) is not None:
        return simulation.get(key)
    return default


def _normalize_path_point(point: Sequence[Any], altitude_hint: float) -> list[float] | None:
    values = _as_point(point, 2)
    if values is None:
        return None
    if len(values) >= 4:
        # Formato corrente: x, y, quota_ft, tempo_sec.
        return [values[0], values[1], values[2], values[3]]
    if len(values) == 3:
        # Formato storico: x, y, tempo. La quota viene dal contesto.
        return [values[0], values[1], altitude_hint, values[2]]
    # Formato puramente 2D.
    return [values[0], values[1], altitude_hint, 0.0]


def path_da_segmenti(
    segments: Sequence[Mapping[str, Any]], altitude_hint: float,
) -> list[list[float]]:
    if not segments:
        return []
    first = segments[0]
    p0 = _as_point(first.get("p0", first.get("start")), 2)
    t0 = _as_float(first.get("t0", first.get("start_time_sec", 0.0)), 0.0)
    if p0 is None or t0 is None:
        return []
    altitude = p0[2] if len(p0) >= 3 else altitude_hint
    path = [[p0[0], p0[1], altitude, t0]]
    for segment in segments:
        p1 = _as_point(segment.get("p1", segment.get("end")), 2)
        t1 = _as_float(segment.get("t1", segment.get("end_time_sec")))
        if p1 is None or t1 is None:
            return []
        z1 = p1[2] if len(p1) >= 3 else altitude_hint
        path.append([p1[0], p1[1], z1, t1])
    return path


def path_soluzione(solution: Mapping[str, Any], altitude_hint: float) -> list[list[float]]:
    raw_path = _nested(solution, "path", []) or []
    if isinstance(raw_path, Sequence) and not isinstance(raw_path, (str, bytes)):
        normalized = [
            converted
            for point in raw_path
            if isinstance(point, Sequence) and not isinstance(point, (str, bytes))
            for converted in [_normalize_path_point(point, altitude_hint)]
            if converted is not None
        ]
        if len(normalized) >= 2:
            return normalized
    segments = _nested(solution, "segments", []) or []
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        valid_segments = [item for item in segments if isinstance(item, Mapping)]
        return path_da_segmenti(valid_segments, altitude_hint)
    return []


def _scenario_from_document(document: Mapping[str, Any]) -> dict[str, Any]:
    scenario = document.get("scenario")
    if isinstance(scenario, Mapping):
        return dict(scenario)
    # Compatibilità con vecchi JSON che salvavano lo scenario al top-level.
    return dict(document)


def _population_from_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("population", "pareto_front", "solutions", "candidates"):
        value = document.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _aircraft_from_legacy(scenario: Mapping[str, Any], controlled: bool) -> dict[str, Any] | None:
    if controlled:
        start = scenario.get("ing_late")
        target = scenario.get("usc_late")
        speed = scenario.get("speed_late")
    else:
        start = scenario.get("ing_early")
        target = scenario.get("usc_early")
        speed = scenario.get("speed_early")
    p0 = _as_point(start, 2)
    p_target = _as_point(target, 2)
    if p0 is None or p_target is None:
        return None
    altitude = p0[2] if len(p0) >= 3 else 0.0
    target_altitude = p_target[2] if len(p_target) >= 3 else altitude
    return {
        "p0": [p0[0], p0[1], altitude],
        "p_target": [p_target[0], p_target[1], target_altitude],
        "v0": _as_float(speed, 0.0),
    }


def _aircraft(scenario: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = scenario.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return _aircraft_from_legacy(scenario, controlled=(key == "ac0"))


def _nominal_path(aircraft: Mapping[str, Any]) -> list[list[float]]:
    p0 = _as_point(aircraft.get("p0"), 2)
    target = _as_point(aircraft.get("p_target"), 2)
    if p0 is None or target is None:
        return []
    altitude = p0[2] if len(p0) >= 3 else 0.0
    time_end = _as_float(aircraft.get("exit_time_sec"), 0.0) or 0.0
    return [
        [p0[0], p0[1], altitude, 0.0],
        [target[0], target[1], altitude, time_end],
    ]


def _plot_aircraft(
    ax: Any, aircraft: Mapping[str, Any], color: str, label: str, linestyle: str,
) -> None:
    path = _nominal_path(aircraft)
    if not path:
        return
    ax.plot(
        [point[0] for point in path],
        [point[1] for point in path],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
    )
    ax.scatter(path[0][0], path[0][1], color=color, s=65, zorder=6)
    ax.scatter(path[-1][0], path[-1][1], color=color, marker="X", s=75, zorder=6)


def _solution_index(solution: Mapping[str, Any], fallback: int) -> int:
    try:
        return int(solution.get("solution_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _is_feasible(solution: Mapping[str, Any]) -> bool:
    explicit = _nested(solution, "feasible")
    if explicit is not None:
        return bool(explicit)
    violation = _as_float(_nested(solution, "constraint_violation"), math.inf)
    collision = bool(_nested(solution, "collisione", _nested(solution, "collision", False)))
    return bool(violation is not None and violation <= EPS and not collision)


def _metric(solution: Mapping[str, Any], name: str, fitness_index: int) -> float | None:
    aliases = {
        "path_length": ("path_length_inside",),
        "horizontal_target_error_nm": ("exit_distance", "dist_exit"),
        "final_bearing_error_deg": ("exit_direction", "delta_theta"),
    }
    value = _nested(solution, name)
    if value is None:
        for alias in aliases.get(name, ()):
            value = _nested(solution, alias)
            if value is not None:
                break
    if value is None:
        fitness = solution.get("fitness")
        if isinstance(fitness, Sequence) and len(fitness) > fitness_index:
            value = fitness[fitness_index]
    return _as_float(value)


def _executed_genome(solution: Mapping[str, Any]) -> list[Sequence[Any]]:
    value = _nested(solution, "executed_genome")
    if not value:
        value = solution.get("genome")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [gene for gene in value if isinstance(gene, Sequence) and len(gene) >= 3]


def _annotate_waypoints(
    ax: Any, solution: Mapping[str, Any], path: Sequence[Sequence[float]], index: int,
) -> None:
    genes = _executed_genome(solution)
    # Il path contiene start e uscita; i vertici intermedi corrispondono ai geni eseguiti.
    for waypoint_index, (point, gene) in enumerate(zip(path[1:-1], genes), start=1):
        time_value = _as_float(gene[0], point[3]) or point[3]
        speed_value = _as_float(gene[2], 0.0) or 0.0
        ax.scatter(point[0], point[1], s=18, color="black", zorder=7)
        ax.annotate(
            f"S{index} W{waypoint_index}\n"
            f"t={time_value:.0f}s v={speed_value:.0f}km/h z={point[2]:.0f}ft",
            (point[0], point[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )


def _reference_point(scenario: Mapping[str, Any]) -> Sequence[float] | None:
    reference = scenario.get("reference_point", scenario.get("collision_point"))
    if reference is None:
        ac0 = scenario.get("ac0")
        if isinstance(ac0, Mapping):
            reference = ac0.get("reference_point")
    return _as_point(reference, 2)


def _filter_population(
    population: Sequence[dict[str, Any]],
    solution_indices: set[int] | None,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    ordered = [
        solution for _, solution in sorted(
            enumerate(population),
            key=lambda item: _solution_index(item[1], item[0]),
        )
    ]
    selected = [
        solution
        for position, solution in enumerate(ordered)
        if solution_indices is None or _solution_index(solution, position) in solution_indices
    ]
    return selected if max_candidates is None else selected[:max_candidates]


def _safe_output_path(input_file: str, suffix: str, output_dir: str | None) -> str:
    source = Path(input_file)
    folder = Path(output_dir) if output_dir else source.parent
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / f"{source.stem}{suffix}")


def genera_grafici_esperimento(
    filename: str,
    annotate: bool = True,
    solution_indices: set[int] | None = None,
    max_candidates: int | None = None,
    output_dir: str | None = None,
    show: bool = False,
) -> list[str]:
    try:
        with open(filename, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Impossibile leggere {filename}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{filename}: la radice JSON deve essere un oggetto")

    scenario = _scenario_from_document(document)
    population = _filter_population(
        _population_from_document(document), solution_indices, max_candidates,
    )
    if not population:
        raise ValueError(f"{filename}: nessuna soluzione candidata trovata")

    ac0 = _aircraft(scenario, "ac0")
    ac1 = _aircraft(scenario, "ac1")
    if ac0 is None or ac1 is None:
        raise ValueError(f"{filename}: scenario senza AC0/AC1 riconoscibili")

    p0_ac0 = _as_point(ac0.get("p0"), 2)
    target_ac0 = _as_point(ac0.get("p_target"), 2)
    if p0_ac0 is None or target_ac0 is None:
        raise ValueError(f"{filename}: p0/p_target di AC0 non validi")

    altitude = p0_ac0[2] if len(p0_ac0) >= 3 else _as_float(document.get("fixed_altitude_ft"), 0.0) or 0.0
    area_size = _as_float(scenario.get("area_size", document.get("area_size")), 100.0) or 100.0

    fig, ax = plt.subplots(figsize=(12, 9))

    nfz_label_used = False
    zones = scenario.get("no_fly_zones", [])
    if isinstance(zones, Sequence):
        for zone in zones:
            if not isinstance(zone, Mapping):
                continue
            cx = _as_float(zone.get("center_x"))
            cy = _as_float(zone.get("center_y"))
            radius = _as_float(zone.get("radius_nm"))
            if cx is None or cy is None or radius is None:
                continue
            ax.add_patch(Circle(
                (cx, cy), radius,
                color="crimson",
                alpha=0.18,
                label=None if nfz_label_used else "No-fly zone",
            ))
            nfz_label_used = True
            ax.annotate(
                f"NFZ {zone.get('zone_id', '')}\n"
                f"{zone.get('min_altitude_ft', '-inf')}..{zone.get('max_altitude_ft', '+inf')} ft",
                (cx, cy), fontsize=7, ha="center", va="center",
            )

    _plot_aircraft(ax, ac1, "darkorange", "AC1 intruso", "--")
    backgrounds = scenario.get("background_aircraft", [])
    if isinstance(backgrounds, Sequence):
        for position, background in enumerate(backgrounds, start=1):
            if isinstance(background, Mapping):
                _plot_aircraft(
                    ax, background, "gray",
                    f"Background {background.get('id', position + 1)}", ":",
                )

    ax.scatter(
        p0_ac0[0], p0_ac0[1], color="navy", marker="o", s=90,
        label=f"AC0 start ({altitude:.0f} ft)", zorder=8,
    )
    ax.scatter(
        target_ac0[0], target_ac0[1], color="green", marker="X", s=120,
        label="Target/uscita nominale AC0", zorder=8,
    )

    reference = _reference_point(scenario)
    if reference is not None:
        ax.scatter(
            reference[0], reference[1], color="purple", marker="*", s=130,
            label="Punto storico di conflitto/riferimento", zorder=8,
        )
        separation = _as_float(
            scenario.get("separation_min", document.get("separation_min")), 0.0,
        ) or 0.0
        if separation > 0.0:
            ax.add_patch(Circle(
                (reference[0], reference[1]), separation,
                fill=False, edgecolor="purple", linestyle=":",
                linewidth=1.2, alpha=0.75,
            ))

    colormap = plt.get_cmap("viridis")
    fuzzy_selected = document.get("fuzzy_selected_index")
    fuzzy_selected_int = None
    try:
        fuzzy_selected_int = int(fuzzy_selected) if fuzzy_selected is not None else None
    except (TypeError, ValueError):
        pass

    annotate_candidates = annotate and len(population) <= 20
    feasible_label_used = False
    infeasible_label_used = False
    plotted = 0
    skipped_paths = 0

    for position, solution in enumerate(population):
        path = path_soluzione(solution, altitude)
        if len(path) < 2:
            skipped_paths += 1
            continue
        feasible = _is_feasible(solution)
        color = colormap(position / max(len(population) - 1, 1)) if feasible else "red"
        style = "-" if feasible else "--"
        index = _solution_index(solution, position)
        selected_by_fuzzy = fuzzy_selected_int is not None and index == fuzzy_selected_int
        if selected_by_fuzzy:
            label = f"Scelta fuzzy S{index}"
        elif feasible and not feasible_label_used:
            label, feasible_label_used = "Soluzioni feasible AC0", True
        elif not feasible and not infeasible_label_used:
            label, infeasible_label_used = "Soluzioni NON feasible AC0", True
        else:
            label = None

        ax.plot(
            [point[0] for point in path],
            [point[1] for point in path],
            color=color,
            linestyle=style,
            linewidth=3.0 if selected_by_fuzzy else 1.3,
            alpha=1.0 if selected_by_fuzzy else 0.68,
            label=label,
        )
        ax.scatter(path[-1][0], path[-1][1], color=color, marker="D", s=25, zorder=7)
        plotted += 1
        if annotate_candidates:
            _annotate_waypoints(ax, solution, path, index)
            ax.annotate(
                f"exit S{index}\nt={path[-1][3]:.0f}s z={path[-1][2]:.0f}ft",
                (path[-1][0], path[-1][1]),
                xytext=(4, -18), textcoords="offset points", fontsize=6,
            )

    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"{filename}: nessuna soluzione contiene path/segments visualizzabili")

    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [NM]")
    ax.set_ylabel("Y [NM]")
    ax.set_title(
        f"Candidati AC0 nel piano XY - quota AC0 costante {altitude:.0f} ft\n"
        "La quota e' informativa: non sono presenti salite o discese"
    )
    ax.grid(True, linestyle=":", alpha=0.45)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.tight_layout()
    map_path = _safe_output_path(filename, "_candidates_xy.png", output_dir)
    fig.savefig(map_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show(block=False)
    plt.close(fig)

    fig_obj, ax_obj = plt.subplots(figsize=(9, 6))
    annotate_objectives = annotate and len(population) <= 40
    objective_points = 0
    for position, solution in enumerate(population):
        x_value = _metric(solution, "path_length", 0)
        y_value = _metric(solution, "horizontal_target_error_nm", 1)
        heading_error = _metric(solution, "final_bearing_error_deg", 2)
        if x_value is None or y_value is None:
            continue
        feasible = _is_feasible(solution)
        ax_obj.scatter(
            x_value, y_value,
            c="tab:blue" if feasible else "red",
            marker="o" if feasible else "x",
        )
        objective_points += 1
        if annotate_objectives:
            heading_text = "n/d" if heading_error is None else f"{heading_error:.1f} deg"
            ax_obj.annotate(
                f"S{_solution_index(solution, position)} | dH={heading_text}",
                (x_value, y_value), xytext=(4, 3),
                textcoords="offset points", fontsize=7,
            )

    generated = [map_path]
    if objective_points:
        ax_obj.set_xlabel("Lunghezza percorso [NM]")
        ax_obj.set_ylabel("Errore uscita [NM]")
        ax_obj.set_title("Confronto candidati (errore di prua nelle annotazioni)")
        ax_obj.grid(True, linestyle=":", alpha=0.45)
        fig_obj.tight_layout()
        objective_path = _safe_output_path(filename, "_candidate_objectives.png", output_dir)
        fig_obj.savefig(objective_path, dpi=220, bbox_inches="tight")
        if show:
            plt.show(block=False)
        generated.append(objective_path)
    plt.close(fig_obj)

    if skipped_paths:
        print(f"{filename}: ignorate {skipped_paths} soluzioni senza path valido.")
    return generated


def _discover_files(patterns: Iterable[str]) -> list[str]:
    discovered: set[str] = set()
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        if os.path.isfile(expanded):
            discovered.add(os.path.normpath(expanded))
            continue
        for path in glob.glob(expanded, recursive=True):
            if os.path.isfile(path):
                discovered.add(os.path.normpath(path))

    candidates = sorted(
        path for path in discovered
        if path.lower().endswith(".json")
        and "_infeasible" not in os.path.basename(path)
        and not path.endswith("_pruning_report.json")
    )
    pruned_bases = {
        path[:-len("_pruned.json")]
        for path in candidates
        if path.endswith("_pruned.json")
    }
    return [
        path for path in candidates
        if path.endswith("_pruned.json")
        or not (path.endswith(".json") and path[:-len(".json")] in pruned_bases)
    ]


def _parse_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    try:
        indices = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--indices accetta interi separati da virgole") from exc
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("Gli indici devono essere non negativi")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*",
        help="File o glob JSON. Default: ricerca ricorsiva in data/.",
    )
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--indices", help="Indici solution_index separati da virgole")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--output-dir", help="Cartella in cui salvare i PNG")
    parser.add_argument("--show", action="store_true", help="Mostra anche le figure")
    args = parser.parse_args()

    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates deve essere maggiore di zero")
    try:
        indices = _parse_indices(args.indices)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    patterns = args.paths or [
        "data/**/*_pareto_front_*.json",
        "data/*_pareto_front_*.json",
    ]
    files = _discover_files(patterns)
    if not files:
        print(
            "Nessun fronte Pareto trovato. Passa il file esplicitamente, ad esempio:\n"
            "  python pareto_vis.py data/exp_0_pareto_front_250.json",
            file=sys.stderr,
        )
        return 1

    generated: list[str] = []
    failed = 0
    for path in files:
        try:
            outputs = genera_grafici_esperimento(
                path,
                annotate=not args.no_annotations,
                solution_indices=indices,
                max_candidates=args.max_candidates,
                output_dir=args.output_dir,
                show=args.show,
            )
        except Exception as exc:  # un file corrotto non deve bloccare gli altri
            failed += 1
            print(f"ERRORE {path}: {exc}", file=sys.stderr)
            continue
        generated.extend(outputs)
        print(f"OK {path}: {', '.join(outputs)}")

    print(
        f"Generati {len(generated)} grafici da {len(files) - failed}/{len(files)} file; "
        f"errori={failed}."
    )
    return 0 if generated else 2


if __name__ == "__main__":
    raise SystemExit(main())
