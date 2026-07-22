"""Genera grafici del fronte Pareto e delle traiettorie per ogni scenario.

Output predefinito::

    data/pareto_scenari/scenario_<id>/
        fronte_pareto_3d.png
        traiettorie_2d.png
        traiettorie_3d.png

Esempi::

    python visualizza_pareto_scenari.py
    python visualizza_pareto_scenari.py data/pruned/train/0_pareto_front_250_pruned.json
    python visualizza_pareto_scenari.py --dpi 300 --output-dir risultati
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# Lo script deve funzionare anche su server/WSL senza interfaccia grafica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBJECTIVE_LABELS = (
    "Lunghezza percorso [NM]",
    "Errore orizzontale al target [NM]",
    "Errore di prua finale [deg]",
)


def numero(value: Any) -> float | None:
    """Converte in float soltanto valori numerici finiti."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def scenario_id(path: Path, document: Mapping[str, Any]) -> str:
    """Ricava un identificatore stabile dal JSON o dal nome del file."""
    for container in (document, document.get("scenario", {})):
        if isinstance(container, Mapping):
            for key in ("scenario_id", "id_scenario", "experiment_id"):
                if container.get(key) is not None:
                    return str(container[key])
    match = re.match(r"(?:exp_)?(\d+)_pareto_front", path.stem)
    return match.group(1) if match else path.stem


def popolazione(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("population", "pareto_front", "solutions", "candidates"):
        value = document.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def quota_scenario(document: Mapping[str, Any]) -> float:
    fixed = numero(document.get("fixed_altitude_ft"))
    scenario = document.get("scenario", document)
    if isinstance(scenario, Mapping):
        ac0 = scenario.get("ac0")
        if isinstance(ac0, Mapping):
            p0 = ac0.get("p0")
            if isinstance(p0, Sequence) and len(p0) >= 3:
                altitude = numero(p0[2])
                if altitude is not None:
                    return altitude
    return fixed or 0.0


def traiettoria(solution: Mapping[str, Any], altitude_hint: float) -> list[tuple[float, float, float]]:
    """Legge path correnti (x,y,z,t) e storici (x,y,t oppure x,y)."""
    raw = solution.get("path")
    if raw is None and isinstance(solution.get("sim"), Mapping):
        raw = solution["sim"].get("path")
    points: list[tuple[float, float, float]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for point in raw:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
                continue
            x, y = numero(point[0]), numero(point[1])
            if x is None or y is None:
                continue
            # Nel formato corrente la presenza di quattro campi identifica la quota.
            z = numero(point[2]) if len(point) >= 4 else altitude_hint
            points.append((x, y, altitude_hint if z is None else z))
    return points


def fitness(solution: Mapping[str, Any]) -> tuple[float, float, float] | None:
    raw = solution.get("fitness")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        values = tuple(numero(raw[index]) for index in range(3))
        if all(value is not None for value in values):
            return values  # type: ignore[return-value]

    metrics = solution.get("metriche_reali", solution)
    if not isinstance(metrics, Mapping):
        return None
    values = (
        numero(metrics.get("path_length", metrics.get("path_length_inside"))),
        numero(metrics.get("horizontal_target_error_nm", metrics.get("exit_distance"))),
        numero(metrics.get("final_bearing_error_deg", metrics.get("exit_direction"))),
    )
    return values if all(value is not None for value in values) else None  # type: ignore[return-value]


def aeromobili_scenario(document: Mapping[str, Any]) -> list[tuple[str, list[float], list[float]]]:
    scenario = document.get("scenario", document)
    if not isinstance(scenario, Mapping):
        return []
    aircraft: list[tuple[str, list[float], list[float]]] = []
    for key, label in (("ac0", "Rotta nominale AC0"), ("ac1", "Rotta AC1")):
        ac = scenario.get(key)
        if not isinstance(ac, Mapping):
            continue
        start, target = ac.get("p0"), ac.get("p_target")
        if not isinstance(start, Sequence) or not isinstance(target, Sequence):
            continue
        if len(start) >= 2 and len(target) >= 2:
            altitude = quota_scenario(document)
            p0 = [float(start[0]), float(start[1]), float(start[2]) if len(start) >= 3 else altitude]
            p1 = [float(target[0]), float(target[1]), float(target[2]) if len(target) >= 3 else altitude]
            aircraft.append((label, p0, p1))
    return aircraft


def estremi_ac0(document: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    """Restituisce ingresso e uscita nominale che tutte le soluzioni condividono."""
    for label, start, target in aeromobili_scenario(document):
        if label == "Rotta nominale AC0":
            return start, target
    return None


def file_prunati(data_dir: Path = Path("data/pruned")) -> list[Path]:
    """Scopre soltanto i fronti prodotti dal pruning, inclusi train e test."""
    return sorted(
        path for path in data_dir.rglob("*_pareto_front_*_pruned.json")
        if path.is_file()
    )


def salva_grafici(source: Path, output_root: Path, dpi: int) -> list[Path]:
    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("la radice del JSON non e' un oggetto")

    solutions = popolazione(document)
    if not solutions:
        raise ValueError("nessuna soluzione Pareto trovata")
    sid = scenario_id(source, document)
    destination = output_root / f"scenario_{sid}"
    destination.mkdir(parents=True, exist_ok=True)
    altitude = quota_scenario(document)
    paths = [(index, traiettoria(item, altitude)) for index, item in enumerate(solutions)]
    paths = [(index, path) for index, path in paths if len(path) >= 2]
    endpoints = estremi_ac0(document)
    if endpoints is None:
        raise ValueError("scenario senza p0/p_target validi per AC0")
    ac0_start, ac0_target = endpoints
    invalid_starts = [
        index for index, path in paths
        if not math.isclose(path[0][0], ac0_start[0], abs_tol=1.0e-8)
        or not math.isclose(path[0][1], ac0_start[1], abs_tol=1.0e-8)
    ]
    if invalid_starts:
        raise ValueError(
            "le soluzioni AC0 non condividono il p0 dello scenario: "
            + ", ".join(map(str, invalid_starts[:10]))
        )
    objectives = [(index, fitness(item)) for index, item in enumerate(solutions)]
    objectives = [(index, fit) for index, fit in objectives if fit is not None]
    colors = plt.get_cmap("viridis")
    generated: list[Path] = []

    # Fronte di Pareto nello spazio dei tre obiettivi.
    if objectives:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        values = [fit for _, fit in objectives]
        scatter = ax.scatter(
            [value[0] for value in values], [value[1] for value in values],
            [value[2] for value in values], c=range(len(values)), cmap="viridis",
            s=45, alpha=0.85, edgecolors="black", linewidths=0.35,
        )
        ax.set_xlabel(OBJECTIVE_LABELS[0], labelpad=10)
        ax.set_ylabel(OBJECTIVE_LABELS[1], labelpad=10)
        ax.set_zlabel(OBJECTIVE_LABELS[2], labelpad=10)
        ax.set_title(f"Fronte di Pareto 3D - scenario {sid}")
        fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.7, label="Indice soluzione")
        fig.tight_layout()
        output = destination / "fronte_pareto_3d.png"
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(output)

    if not paths:
        raise ValueError("nessuna traiettoria valida trovata")

    # Traiettorie sul piano XY.
    fig, ax = plt.subplots(figsize=(10, 8))
    for position, (index, path) in enumerate(paths):
        color = colors(position / max(len(paths) - 1, 1))
        ax.plot([p[0] for p in path], [p[1] for p in path], color=color, alpha=0.62, linewidth=1.4)
        ax.scatter(path[-1][0], path[-1][1], color=color, s=12)
    ax.scatter(ac0_start[0], ac0_start[1], color="navy", marker="o", s=90,
               label="Partenza comune AC0", zorder=8)
    ax.scatter(ac0_target[0], ac0_target[1], color="green", marker="X", s=120,
               label="Uscita nominale AC0", zorder=8)
    for label, start, target in aeromobili_scenario(document):
        ax.plot([start[0], target[0]], [start[1], target[1]], "--", linewidth=2, label=label)
    ax.set_xlabel("X [NM]")
    ax.set_ylabel("Y [NM]")
    ax.set_title(f"Traiettorie Pareto 2D - scenario {sid}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    if aeromobili_scenario(document):
        ax.legend()
    fig.tight_layout()
    output = destination / "traiettorie_2d.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    generated.append(output)

    # Traiettorie nello spazio XYZ (quota in piedi).
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for position, (index, path) in enumerate(paths):
        color = colors(position / max(len(paths) - 1, 1))
        ax.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path],
                color=color, alpha=0.65, linewidth=1.4)
    ax.scatter(ac0_start[0], ac0_start[1], ac0_start[2], color="navy", marker="o",
               s=90, label="Partenza comune AC0")
    ax.scatter(ac0_target[0], ac0_target[1], ac0_target[2], color="green", marker="X",
               s=120, label="Uscita nominale AC0")
    for label, start, target in aeromobili_scenario(document):
        ax.plot([start[0], target[0]], [start[1], target[1]], [start[2], target[2]],
                "--", linewidth=2.2, label=label)
    ax.set_xlabel("X [NM]")
    ax.set_ylabel("Y [NM]")
    ax.set_zlabel("Quota [ft]")
    ax.set_title(f"Traiettorie Pareto 3D - scenario {sid}")
    if aeromobili_scenario(document):
        ax.legend()
    fig.tight_layout()
    output = destination / "traiettorie_3d.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    generated.append(output)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="JSON pruned da elaborare (default: data/pruned/{train,test}/**/*_pruned.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/pareto_scenari"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi deve essere positivo")

    files = args.files or file_prunati()
    if not files:
        print("Nessun file Pareto trovato.", file=sys.stderr)
        return 1

    failures = 0
    for source in files:
        try:
            outputs = salva_grafici(source, args.output_dir, args.dpi)
            print(f"OK {source}: {len(outputs)} grafici in {outputs[0].parent}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"ERRORE {source}: {exc}", file=sys.stderr)
    print(f"Completati {len(files) - failures}/{len(files)} scenari; errori: {failures}.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
