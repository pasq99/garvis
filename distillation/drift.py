"""Empirical and readable drift between two quadratic TSK surrogates."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from distillation.data import load_fronts
from distillation.fuzzy import FuzzyBasis, RULES, TSK_TERMS, basis_signature


GRID_LEVELS = 17
TOP_RULES = 10
PERTURBATIONS = (
    ("path", 0, 0.05 / 1.5, "rapporto percorso +0.05"),
    ("horizontal", 1, 0.01 / 0.10, "errore orizzontale +0.01 NM"),
    ("bearing", 2, 5.0 / 180.0, "errore bearing +5°"),
)
TERM_LABELS = {
    "direct": "diretto", "moderate": "moderato", "long": "lungo",
    "precise": "preciso", "acceptable": "accettabile", "limit": "al limite",
    "aligned": "allineato", "deviated": "deviato", "poor": "scarso",
}


def _load(path: str | Path) -> tuple[dict, np.ndarray]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("type") not in {"tsk_surrogate", "fuzzy_surrogate"}:
        raise ValueError(f"Artefatto fuzzy non valido: {path}")
    if artifact.get("basis_signature") != basis_signature():
        raise ValueError(f"Base fuzzy incompatibile: {path}")
    quadratic = (artifact.get("models") or {}).get("quadratic")
    if not quadratic or tuple(quadratic.get("terms") or ()) != TSK_TERMS:
        raise ValueError(f"TSK quadratico incompatibile: {path}")
    coefficients = np.asarray(quadratic.get("coefficients"), dtype=np.float64)
    if coefficients.shape != (len(RULES), len(TSK_TERMS)) or not np.isfinite(coefficients).all():
        raise ValueError(f"Coefficienti TSK non validi: {path}")
    return artifact, coefficients


def _weighted(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total else None


def _effect(
    basis: FuzzyBasis, features: np.ndarray, coefficients: np.ndarray, column: int, step: float
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(features[:, column] <= 1.0 - step)
    shifted = features[indices].copy()
    shifted[:, column] += step
    return indices, basis.score(shifted, coefficients) - basis.score(features[indices], coefficients)


def _fidelity(artifact: dict) -> dict:
    metrics = artifact["models"]["quadratic"].get("test") or artifact.get("selected_test") or {}
    return {key: metrics.get(key) for key in (
        "top1_fidelity", "top2_fidelity", "pairwise_agreement", "rmse",
    )}


def _antecedent(rule: tuple[str, str, str]) -> str:
    path, horizontal, bearing = rule
    return (
        f"percorso {TERM_LABELS[path]}, orizzontale {TERM_LABELS[horizontal]}, "
        f"bearing {TERM_LABELS[bearing]}"
    )


def _sensitivity_presentation(before: float, after: float) -> dict:
    if before < 0.0 and after < 0.0:
        first, second = abs(before), abs(after)
        return {
            "kind": "penalità", "before": first, "after": second,
            "relative_change_pct": 100.0 * (second / first - 1.0),
            "interpretation": (
                "maggiore tolleranza" if second < first else "minore tolleranza"
            ),
        }
    if before > 0.0 and after > 0.0:
        return {
            "kind": "premio", "before": before, "after": after,
            "relative_change_pct": 100.0 * (after / before - 1.0),
            "interpretation": "premio ridotto" if after < before else "premio aumentato",
        }
    return {
        "kind": "effetto", "before": before, "after": after,
        "relative_change_pct": None, "interpretation": "segno invertito",
    }


def generate_drift(
    before_path: str | Path,
    after_path: str | Path,
    data_folder: str | Path,
    output: str | Path,
) -> Path:
    before, before_coefficients = _load(before_path)
    after, after_coefficients = _load(after_path)
    if before.get("features") != after.get("features"):
        raise ValueError("I surrogate usano feature differenti")

    fronts = load_fronts(data_folder)
    features = np.concatenate([front.features for front in fronts]).astype(np.float64)
    sizes = np.asarray([len(front.features) for front in fronts], dtype=np.int32)
    basis = FuzzyBasis()
    activation = basis.activation(features)
    before_score = basis.score(features, before_coefficients)
    after_score = basis.score(features, after_coefficients)
    relative_delta = after_score - after_score.mean() - before_score + before_score.mean()

    changed = offset = 0
    for size in sizes:
        size = int(size)
        changed += np.argmax(before_score[offset:offset + size]) != np.argmax(
            after_score[offset:offset + size]
        )
        offset += size

    perturbations = []
    effect_arrays = {}
    for name, column, step, label in PERTURBATIONS:
        indices, first = _effect(basis, features, before_coefficients, column, step)
        after_indices, second = _effect(basis, features, after_coefficients, column, step)
        assert np.array_equal(indices, after_indices)
        effect_arrays[name] = (indices, first, second)
        first_mean, second_mean = float(first.mean()), float(second.mean())
        presentation = _sensitivity_presentation(first_mean, second_mean)
        perturbations.append({
            "key": name, "label": label, "samples": int(len(indices)),
            "before": first_mean, "after": second_mean,
            "change": second_mean - first_mean,
            "presentation": presentation,
        })

    dominant = np.argmax(activation, axis=1)
    coefficient_delta = after_coefficients - before_coefficients
    rules = []
    for index, rule in enumerate(RULES):
        weights = activation[:, index]
        effects = {}
        for name, _, _, _ in PERTURBATIONS:
            indices, first, second = effect_arrays[name]
            before_effect = _weighted(first, weights[indices])
            after_effect = _weighted(second, weights[indices])
            effects[name] = {
                "before": before_effect, "after": after_effect,
                "change": (
                    after_effect - before_effect
                    if before_effect is not None and after_effect is not None else None
                ),
            }
        order = np.argsort(-np.abs(coefficient_delta[index]))[:3]
        rules.append({
            "rule": index + 1,
            "antecedent": {"path": rule[0], "horizontal": rule[1], "bearing": rule[2]},
            "label": _antecedent(rule),
            "support": float(weights.mean()),
            "active_routes": int((weights > 0.0).sum()),
            "dominant_routes": int((dominant == index).sum()),
            "mean_relative_drift": _weighted(relative_delta, weights),
            "intensity": _weighted(np.abs(relative_delta), weights),
            "effects": effects,
            "top_coefficient_changes": [
                {"term": TSK_TERMS[term], "delta": float(coefficient_delta[index, term])}
                for term in order
            ],
        })
    rules.sort(key=lambda item: item["intensity"] if item["intensity"] is not None else -1.0, reverse=True)
    for rank, rule in enumerate(rules, 1):
        rule["rank"] = rank

    axis = np.linspace(0.0, 1.0, GRID_LEVELS)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_activation = basis.activation(grid)
    grid_before = basis.score(grid, before_coefficients)
    grid_after = basis.score(grid, after_coefficients)
    grid_delta = grid_after - grid_after.mean() - grid_before + grid_before.mean()
    grid_rules = sorted((
        (_weighted(np.abs(grid_delta), grid_activation[:, index]), index)
        for index in range(len(RULES))
    ), reverse=True)

    result = {
        "before": {
            "surrogate": str(before_path), "model_sha256": before.get("model_sha256"),
            "fidelity": _fidelity(before),
        },
        "after": {
            "surrogate": str(after_path), "model_sha256": after.get("model_sha256"),
            "fidelity": _fidelity(after),
        },
        "reference": {"data": str(data_folder), "fronts": len(fronts), "routes": len(features)},
        "empirical": {
            "score_correlation": float(np.corrcoef(before_score, after_score)[0, 1]),
            "mean_absolute_drift": float(np.mean(np.abs(relative_delta))),
            "rms_drift": float(np.sqrt(np.mean(relative_delta ** 2))),
            "p95_absolute_drift": float(np.percentile(np.abs(relative_delta), 95)),
            "top1_agreement": 1.0 - changed / len(fronts),
            "changed_fronts": int(changed),
            "perturbations": perturbations,
            "rules": rules,
        },
        "uniform_grid_appendix": {
            "levels": GRID_LEVELS, "points": len(grid),
            "mean_absolute_drift": float(np.mean(np.abs(grid_delta))),
            "rms_drift": float(np.sqrt(np.mean(grid_delta ** 2))),
            "p95_absolute_drift": float(np.percentile(np.abs(grid_delta), 95)),
            "top_rules": [
                {"rule": index + 1, "label": _antecedent(RULES[index]), "intensity": intensity}
                for intensity, index in grid_rules[:5]
            ],
        },
    }

    destination = Path(output)
    data_destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data_destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    empirical = result["empirical"]
    lines = [
        "# Drift globale PPO⁰ → PPO fine-tuned",
        "",
        f"> Analisi empirica su **{len(fronts)} fronti** e **{len(features)} rotte** da `{data_folder}`.",
        "",
        "## Sintesi esecutiva",
        "",
        "| Indicatore | Risultato |",
        "|---|---:|",
        f"| Accordo top-1 tra surrogate | **{empirical['top1_agreement']:.2%}** |",
        f"| Fronti con decisione diversa | **{changed}/{len(fronts)}** |",
        f"| Correlazione dei punteggi | **{empirical['score_correlation']:.4f}** |",
        f"| Drift assoluto medio | **{empirical['mean_absolute_drift']:.4f}** |",
        f"| Drift RMS | **{empirical['rms_drift']:.4f}** |",
        f"| P95 del drift assoluto | **{empirical['p95_absolute_drift']:.4f}** |",
        "",
        "### Fidelity della spiegazione",
        "",
        "| Surrogate | Top-1 | Top-2 | Pairwise | RMSE logit |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, artifact in (("PPO⁰", result["before"]), ("Fine-tuned", result["after"])):
        fidelity = artifact["fidelity"]
        lines.append(
            f"| {label} | {fidelity['top1_fidelity']:.2%} | {fidelity['top2_fidelity']:.2%} | "
            f"{fidelity['pairwise_agreement']:.2%} | {fidelity['rmse']:.3f} |"
        )

    lines.extend([
        "",
        "## Cosa è cambiato nel comportamento",
        "",
        "Per ogni esperimento peggioriamo una sola caratteristica della stessa rotta. Una penalità più piccola significa che il modello fine-tuned è diventato più tollerante.",
        "",
        "| Peggioramento simulato | PPO⁰ | Fine-tuned | Variazione | Significato |",
        "|---|---:|---:|---:|---|",
    ])
    for item in perturbations:
        view = item["presentation"]
        variation = (
            f"{view['relative_change_pct']:+.1f}% di {view['kind']}"
            if view["relative_change_pct"] is not None else "n/d"
        )
        lines.append(
            f"| {item['label']} | {view['kind']} **{view['before']:.2f}** | "
            f"{view['kind']} **{view['after']:.2f}** | {variation} | "
            f"**{view['interpretation']}** |"
        )

    lines.extend([
        "",
        f"## Le {TOP_RULES} regioni fuzzy con il maggiore cambiamento locale",
        "",
        "Il cambiamento è espresso in punti del surrogate, non in percentuale o probabilità. Un supporto basso indica che la regione compare raramente.",
        "",
        "| Pos. | Regola | Regione fuzzy | Cambiamento medio | Supporto | Rotte dominanti |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for rule in rules[:TOP_RULES]:
        mean_drift = rule["mean_relative_drift"]
        change = (
            f"{abs(mean_drift):.2f} punti, "
            f"{'più favorita' if mean_drift > 0 else 'meno favorita'}"
            if mean_drift is not None else "supporto assente"
        )
        lines.append(
            f"| {rule['rank']} | R{rule['rule']} | {rule['label']} | "
            f"**{change}** | {rule['support']:.3%} | "
            f"{rule['dominant_routes']} |"
        )

    grid_summary = result["uniform_grid_appendix"]
    lines.extend([
        "",
        "## Appendice — esplorazione fuori distribuzione",
        "",
        f"La griglia uniforme {GRID_LEVELS}³ non rappresenta la frequenza reale delle rotte. È mantenuta solo per individuare extrapolazioni del TSK.",
        "",
        "| Indicatore | Griglia uniforme |",
        "|---|---:|",
        f"| Drift assoluto medio | {grid_summary['mean_absolute_drift']:.4f} |",
        f"| Drift RMS | {grid_summary['rms_drift']:.4f} |",
        f"| P95 assoluto | {grid_summary['p95_absolute_drift']:.4f} |",
        "",
        "| # | Regione estrema | Intensità griglia |",
        "|---:|---|---:|",
        *(
            f"| {rule['rule']} | {rule['label']} | {rule['intensity']:.4f} |"
            for rule in grid_summary["top_rules"]
        ),
        "",
        "## Limiti",
        "",
        "- Il drift spiega i surrogate, non regole interne realmente possedute dal PPO.",
        "- Le fidelity riportate delimitano l'affidabilità dell'approssimazione.",
        "- Le perturbazioni isolate possono uscire dalla distribuzione congiunta delle feature.",
        f"- I dati completi delle 27 regole sono in [`{data_destination.name}`]({data_destination.name}).",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def print_summary(json_path: str | Path) -> None:
    result = json.loads(Path(json_path).read_text(encoding="utf-8"))
    empirical, reference = result["empirical"], result["reference"]
    print("\nDRIFT FUZZY — SINTESI EMPIRICA")
    print(f"Riferimento : {reference['fronts']} fronti, {reference['routes']} rotte")
    print(
        f"Top-1       : {empirical['top1_agreement']:.2%} accordo "
        f"({empirical['changed_fronts']} fronti cambiati)"
    )
    print(f"Decisioni   : {empirical['changed_fronts']} fronti su {reference['fronts']} sono cambiati")
    print("\nTOLLERANZA AI PEGGIORAMENTI")
    for item in empirical["perturbations"]:
        view = item["presentation"]
        variation = (
            f"{view['relative_change_pct']:+.1f}%"
            if view["relative_change_pct"] is not None else "n/d"
        )
        print(f"  {item['label']}")
        print(
            f"    {view['kind']}: {view['before']:.2f} -> {view['after']:.2f} "
            f"({variation}) | {view['interpretation']}"
        )
    print("\nREGIONI FUZZY PIÙ CAMBIATE")
    for rule in empirical["rules"][:5]:
        mean_drift = rule["mean_relative_drift"]
        direction = "più favorita" if mean_drift > 0 else "meno favorita"
        print(
            f"  {rule['rank']}. R{rule['rule']} | {direction} di {abs(mean_drift):.2f} punti | "
            f"supporto {rule['support']:.3%}\n"
            f"     {rule['label']}"
        )


def self_check() -> None:
    before = np.zeros((len(RULES), len(TSK_TERMS)))
    before[:, 1] = 1.0
    after = before.copy()
    after[0, 0] = 1.0
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        before_path, after_path, output = root / "before.json", root / "after.json", root / "drift.md"
        for path, coefficients in ((before_path, before), (after_path, after)):
            path.write_text(json.dumps({
                "type": "fuzzy_surrogate", "basis_signature": basis_signature(),
                "features": ["path", "horizontal", "bearing"],
                "model_sha256": path.stem, "models": {"quadratic": {
                    "terms": list(TSK_TERMS), "coefficients": coefficients.tolist(),
                    "test": {"top1_fidelity": 1.0, "top2_fidelity": 1.0,
                             "pairwise_agreement": 1.0, "rmse": 0.0},
                }},
            }), encoding="utf-8")
        data = root / "data"
        data.mkdir()
        (data / "1_pareto_front_1_pruned.json").write_text(json.dumps({
            "scenario": {"ac0": {"p0": [0, 0], "p_target": [1, 0]}},
            "population": [
                {"path_length": 1.0, "horizontal_target_error_nm": 0.0,
                 "final_bearing_error_deg": 0.0},
                {"path_length": 1.5, "horizontal_target_error_nm": 0.02,
                 "final_bearing_error_deg": 5.0},
            ],
        }), encoding="utf-8")
        generate_drift(before_path, after_path, data, output)
        report = output.read_text(encoding="utf-8")
        result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        assert "Sintesi esecutiva" in report and len(result["empirical"]["rules"]) == 27


if __name__ == "__main__":
    self_check()
