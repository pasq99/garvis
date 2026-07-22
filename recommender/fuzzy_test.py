"""Valuta e visualizza l'oracolo fuzzy sui fronti Pareto potati."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RECOMMENDER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RECOMMENDER_DIR.parent
for module_path in (RECOMMENDER_DIR, PROJECT_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from rlfuzzy import FuzzyDecisionMaker, dist_diretta_2d


def trova_file_pruned(data_dir: str | Path) -> list[Path]:
    """Trova tutti i fronti potati, inclusi gli split train/test."""
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Directory pruned non trovata: {root}")
    return sorted(
        path for path in root.rglob("*_pareto_front_*_pruned.json")
        if "_infeasible" not in path.name
    )


def valuta_file(path: Path, fuzzy: FuzzyDecisionMaker, data_root: Path) -> dict[str, Any]:
    """Calcola il voto fuzzy di ogni soluzione di un fronte potato."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    scenario = document.get("scenario") or {}
    ac0 = scenario.get("ac0") or {}
    direct_distance = dist_diretta_2d(ac0.get("p0"), ac0.get("p_target"))
    if direct_distance is None or direct_distance <= 0.0:
        raise ValueError(f"Distanza diretta non calcolabile in {path}")

    solutions: list[dict[str, Any]] = []
    for fallback_index, candidate in enumerate(document.get("population") or []):
        path_length, horizontal_error, bearing_error = fuzzy._extract_solution_metrics(candidate)
        path_ratio = max(1.0, path_length / direct_distance)
        preference = fuzzy.evaluate_solution(
            path_length,
            horizontal_error,
            bearing_error,
            dist_diretta=direct_distance,
        )
        solutions.append({
            "solution_index": int(candidate.get("solution_index", fallback_index)),
            "source_solution_index": candidate.get("source_solution_index"),
            "path_length_nm": path_length,
            "direct_distance_nm": direct_distance,
            "path_ratio": path_ratio,
            "horizontal_target_error_nm": horizontal_error,
            "final_bearing_error_deg": bearing_error,
            "fuzzy_preference": preference,
        })

    if not solutions:
        raise ValueError(f"Fronte Pareto vuoto: {path}")

    scores = np.asarray([item["fuzzy_preference"] for item in solutions], dtype=float)
    ranking = np.argsort(-scores, kind="stable").astype(int).tolist()
    split = path.relative_to(data_root).parts[0] if path.parent != data_root else "pruned"
    return {
        "file": str(path),
        "filename": path.name,
        "split": split,
        "num_solutions": len(solutions),
        "best_solution_index": solutions[ranking[0]]["solution_index"],
        "best_fuzzy_preference": float(scores[ranking[0]]),
        "ranking": [solutions[index]["solution_index"] for index in ranking],
        "solutions": solutions,
    }


def plot_membership_functions(fuzzy: FuzzyDecisionMaker, output_dir: Path) -> Path:
    variables = (
        (fuzzy.path_ratio, "Rapporto percorso / distanza diretta"),
        (fuzzy.horizontal_err, "Errore orizzontale al target [NM]"),
        (fuzzy.bearing_err, "Errore di prua finale [gradi]"),
        (fuzzy.preference, "Preferenza fuzzy"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (variable, title) in zip(axes.flat, variables):
        for name, term in variable.terms.items():
            axis.plot(variable.universe, term.mf, label=name, linewidth=2)
        axis.set_title(title)
        axis.set_ylabel("Grado di appartenenza")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    output = output_dir / "membership_functions.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_scenario(report: dict[str, Any], output_dir: Path) -> Path:
    """Grafico delle metriche e dei voti di tutte le soluzioni di uno scenario."""
    solutions = report["solutions"]
    indices = np.asarray([item["solution_index"] for item in solutions], dtype=int)
    ratios = np.asarray([item["path_ratio"] for item in solutions], dtype=float)
    horizontal = np.asarray([item["horizontal_target_error_nm"] for item in solutions], dtype=float)
    bearing = np.asarray([item["final_bearing_error_deg"] for item in solutions], dtype=float)
    preferences = np.asarray([item["fuzzy_preference"] for item in solutions], dtype=float)
    best_position = int(np.argmax(preferences))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(indices, preferences, color="tab:blue", alpha=0.8)
    axes[0, 0].bar(indices[best_position], preferences[best_position], color="tab:green")
    axes[0, 0].set(title="Voto fuzzy per soluzione", xlabel="Indice soluzione", ylabel="Preferenza [0-100]")
    axes[0, 0].set_ylim(0, 100)

    scatter = axes[0, 1].scatter(ratios, horizontal, c=preferences, cmap="viridis", vmin=0, vmax=100, s=65)
    axes[0, 1].scatter(ratios[best_position], horizontal[best_position], marker="*", s=240,
                       facecolors="none", edgecolors="red", linewidths=1.8)
    axes[0, 1].set(title="Percorso ed errore al target", xlabel="Path ratio", ylabel="Errore target [NM]")
    fig.colorbar(scatter, ax=axes[0, 1], label="Preferenza fuzzy")

    axes[1, 0].scatter(bearing, preferences, color="tab:orange", s=55)
    for index, x_value, y_value in zip(indices, bearing, preferences):
        axes[1, 0].annotate(str(index), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[1, 0].set(title="Effetto dell'allineamento finale", xlabel="Errore prua [gradi]", ylabel="Preferenza")

    metric_matrix = np.column_stack((
        (ratios - 1.0) / 1.5,
        horizontal / 0.10,
        bearing / 180.0,
    ))
    x = np.arange(len(indices))
    width = 0.25
    labels = ("Path ratio", "Errore target", "Errore prua")
    for column, label in enumerate(labels):
        axes[1, 1].bar(x + (column - 1) * width, metric_matrix[:, column], width, label=label)
    axes[1, 1].set(title="Input fuzzy normalizzati", xlabel="Posizione nel fronte", ylabel="Valore normalizzato")
    axes[1, 1].set_xticks(x, indices)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle(
        f"{report['split']}/{report['filename']} - migliore: {report['best_solution_index']} "
        f"({report['best_fuzzy_preference']:.2f})"
    )
    fig.tight_layout()
    scenario_dir = output_dir / "scenarios" / report["split"]
    scenario_dir.mkdir(parents=True, exist_ok=True)
    output = scenario_dir / f"{Path(report['filename']).stem}_fuzzy.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_aggregate(reports: list[dict[str, Any]], output_dir: Path) -> Path:
    scores = [score for report in reports for score in
              (item["fuzzy_preference"] for item in report["solutions"])]
    best_scores = [report["best_fuzzy_preference"] for report in reports]
    counts = [report["num_solutions"] for report in reports]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(scores, bins=np.linspace(0, 100, 21), color="tab:blue", alpha=0.8)
    axes[0].set(title="Voti di tutte le soluzioni", xlabel="Preferenza fuzzy", ylabel="Frequenza")
    axes[1].hist(best_scores, bins=np.linspace(0, 100, 21), color="tab:green", alpha=0.8)
    axes[1].set(title="Miglior voto per scenario", xlabel="Preferenza fuzzy", ylabel="Scenari")
    axes[2].scatter(counts, best_scores, color="tab:purple", alpha=0.8)
    axes[2].set(title="Dimensione fronte e miglior voto", xlabel="Numero soluzioni", ylabel="Miglior preferenza")
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    output = output_dir / "fuzzy_aggregate.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default="data/pruned")
    parser.add_argument("--output-dir", default="recommender/data/fuzzy_plots")
    parser.add_argument("--no-scenario-plots", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = trova_file_pruned(data_root)
    if not files:
        print(f"Nessun fronte Pareto potato trovato in {data_root}")
        return 1

    fuzzy = FuzzyDecisionMaker()
    reports: list[dict[str, Any]] = []
    for path in files:
        try:
            report = valuta_file(path, fuzzy, data_root)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[IGNORATO] {path}: {exc}")
            continue
        reports.append(report)
        print(
            f"[{report['split']}] {report['filename']}: soluzioni={report['num_solutions']}, "
            f"migliore={report['best_solution_index']}, voto={report['best_fuzzy_preference']:.2f}"
        )
        if not args.no_scenario_plots:
            plot_scenario(report, output_dir)

    if not reports:
        print("Nessuno scenario valutabile")
        return 2

    membership_path = plot_membership_functions(fuzzy, output_dir)
    aggregate_path = plot_aggregate(reports, output_dir)
    report_path = output_dir / "fuzzy_scores.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    total_solutions = sum(report["num_solutions"] for report in reports)
    print(f"\nValutati {len(reports)} scenari e {total_solutions} soluzioni.")
    print(f"Membership: {membership_path}")
    print(f"Riepilogo: {aggregate_path}")
    print(f"Voti completi: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
