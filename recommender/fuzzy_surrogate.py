"""Extract an auditable fuzzy surrogate from the contextual PPO policy.

The PPO checkpoint is treated as a frozen oracle.  This module never trains or
changes the policy: it records its ranking on complete Pareto fronts and fits
only the consequents of a 27-rule fuzzy system whose antecedent membership
functions are inherited from :mod:`rlfuzzy`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

import numpy as np
import skfuzzy as fuzz
import torch as th
from torch import nn
from sb3_contrib import MaskablePPO

from rlfuzzy import (
    FuzzyDecisionMaker,
    MATCH_TOLERANCE,
    MultiScenarioEnv,
    ROUTE_FEATURES,
    split_train_validation,
)


PATH_TERMS = ("direct", "moderate", "long")
HORIZONTAL_TERMS = ("precise", "acceptable", "limit")
BEARING_TERMS = ("aligned", "deviated", "poor")
OUTPUT_TERMS = ("very_low", "low", "medium", "high", "optimal")
RULES = tuple(
    (path, horizontal, bearing)
    for path in PATH_TERMS
    for horizontal in HORIZONTAL_TERMS
    for bearing in BEARING_TERMS
)
DEFAULT_MODEL = "results_multiseed/seed_46/best_model/best_model.zip"
DEFAULT_DATA = "data/pruned/train"
DEFAULT_OUTPUT = "results_surrogate"
EPSILON = 1e-8


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File richiesto non trovato: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation(filename: str) -> int:
    match = re.search(r"_pareto_front_(\d+)", filename)
    return int(match.group(1)) if match else -1


def _rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


@dataclass(frozen=True)
class FuzzyBasis:
    """Fixed antecedent grid and output centroids from the initial FIS."""

    fuzzy: FuzzyDecisionMaker

    @property
    def output_centroids(self) -> np.ndarray:
        universe = self.fuzzy.preference.universe
        return np.asarray([
            fuzz.defuzz(universe, self.fuzzy.preference[name].mf, "centroid")
            for name in OUTPUT_TERMS
        ], dtype=np.float32)

    def activation(self, features: np.ndarray, *, operator: str = "product") -> np.ndarray:
        """Return memberships for all 27 atomic rules.

        ``features`` are the normalized route features used by PPO.  The
        original physical domains are reconstructed before evaluating the
        frozen membership functions.
        """
        array = np.asarray(features, dtype=np.float64)
        if array.shape[-1] != len(ROUTE_FEATURES):
            raise ValueError(f"Attese {len(ROUTE_FEATURES)} feature, ricevute {array.shape}")
        original_shape = array.shape[:-1]
        flat = array.reshape(-1, len(ROUTE_FEATURES))
        values = (
            1.0 + 1.5 * np.clip(flat[:, 0], 0.0, 1.0),
            0.10 * np.clip(flat[:, 1], 0.0, 1.0),
            180.0 * np.clip(flat[:, 2], 0.0, 1.0),
        )
        variables = (self.fuzzy.path_ratio, self.fuzzy.horizontal_err, self.fuzzy.bearing_err)
        term_groups = (PATH_TERMS, HORIZONTAL_TERMS, BEARING_TERMS)
        memberships: list[dict[str, np.ndarray]] = []
        for variable, names, column in zip(variables, term_groups, values):
            memberships.append({
                name: np.asarray([
                    fuzz.interp_membership(variable.universe, variable[name].mf, value)
                    for value in column
                ], dtype=np.float64)
                for name in names
            })
        activations = []
        for path, horizontal, bearing in RULES:
            components = (
                memberships[0][path], memberships[1][horizontal], memberships[2][bearing]
            )
            if operator == "product":
                activation = components[0] * components[1] * components[2]
            elif operator == "min":
                activation = np.minimum(np.minimum(components[0], components[1]), components[2])
            else:
                raise ValueError("operator deve essere 'product' oppure 'min'")
            activations.append(activation)
        result = np.stack(activations, axis=-1).astype(np.float32)
        return result.reshape(*original_shape, len(RULES))


def _representative_values() -> dict[str, float]:
    # Plateau terms use a point inside both the fuzzy core and the empirical
    # domain of the current pruned fronts.
    return {
        "direct": 1.015,
        "moderate": 1.15,
        "long": 1.40,
        "precise": 0.0025,
        "acceptable": 0.040,
        "limit": 0.0925,
        "aligned": 0.50,
        "deviated": 10.0,
        "poor": 45.0,
    }


def _representative_feature(rule: tuple[str, str, str]) -> np.ndarray:
    values = _representative_values()
    return np.asarray([
        (values[rule[0]] - 1.0) / 1.5,
        values[rule[1]] / 0.10,
        values[rule[2]] / 180.0,
    ], dtype=np.float32)


def _initial_rule_labels(basis: FuzzyBasis) -> tuple[list[str], np.ndarray]:
    centroids = basis.output_centroids
    labels: list[str] = []
    values: list[float] = []
    representatives = _representative_values()
    for path, horizontal, bearing in RULES:
        score = basis.fuzzy.evaluate_solution(
            representatives[path], representatives[horizontal], representatives[bearing], 1.0
        )
        index = int(np.argmin(np.abs(centroids - score)))
        labels.append(OUTPUT_TERMS[index])
        values.append(float(centroids[index]))
    return labels, np.asarray(values, dtype=np.float32)


def _features_to_observation(features: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) > capacity:
        raise ValueError(f"Fronte da {len(features)} rotte, capacita' PPO {capacity}")
    padded = np.full((capacity, len(ROUTE_FEATURES)), -1.0, dtype=np.float32)
    padded[:len(features)] = features
    mask = np.zeros(capacity, dtype=bool)
    mask[:len(features)] = True
    return padded.reshape(-1), mask


def _policy_outputs(
    model: MaskablePPO, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    observation, mask = _features_to_observation(features, int(model.action_space.n))
    device = model.device
    observation_tensor = th.as_tensor(observation, dtype=th.float32, device=device).unsqueeze(0)
    mask_tensor = th.as_tensor(mask, dtype=th.bool, device=device).unsqueeze(0)
    with th.no_grad():
        distribution = model.policy.get_distribution(
            observation_tensor, action_masks=mask_tensor
        ).distribution
        logits = distribution.logits[0, :len(features)].detach().cpu().numpy().astype(float)
        probabilities = distribution.probs[0, :len(features)].detach().cpu().numpy().astype(float)
    selected = int(np.argmax(probabilities))
    return logits, probabilities, selected


def _split_scenarios(
    scenarios: Sequence[dict[str, Any]], seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    identification, audit = split_train_validation(scenarios, 0.20, seed)
    fit, calibration = split_train_validation(identification, 0.20, seed + 1)
    split_map: dict[str, str] = {}
    for name, subset in (("fit", fit), ("calibration", calibration), ("audit", audit)):
        for scenario in subset:
            split_map[scenario["group_id"]] = name
    groups = {
        name: sorted(group for group, split in split_map.items() if split == name)
        for name in ("fit", "calibration", "audit")
    }
    manifest = {
        "seed": seed,
        "strategy": "grouped_by_physical_scenario_and_stratified_by_front_size",
        "groups": groups,
        "num_groups": {name: len(value) for name, value in groups.items()},
        "num_fronts": {
            name: sum(split_map[item["group_id"]] == name for item in scenarios)
            for name in groups
        },
    }
    return split_map, manifest


def extract_policy_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint PPO non trovato: {model_path}")
    model = MaskablePPO.load(str(model_path), device=args.device)
    fuzzy_decider = FuzzyDecisionMaker()
    env = MultiScenarioEnv(
        args.data_folder,
        fuzzy_decider,
        max_pareto_size=int(model.action_space.n),
        scenario_sampling="cycle",
        permute_candidates=False,
    )
    split_map, manifest = _split_scenarios(env.scenarios, args.split_seed)
    manifest.update({
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "data_folder": args.data_folder,
        "capacity": int(model.action_space.n),
    })
    _json_dump(output_dir / "split_manifest.json", manifest)
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    permutation_deltas: list[float] = []
    permutation_matches: list[bool] = []
    for scenario in env.scenarios:
        features = np.asarray(scenario["policy_features"], dtype=np.float32)
        logits, probabilities, selected = _policy_outputs(model, features)
        routes = []
        for index, (solution, feature) in enumerate(zip(scenario["pareto_front"], features)):
            routes.append({
                "source_idx": int(solution["source_idx"]),
                "path_ratio": float(1.0 + 1.5 * feature[0]),
                "horizontal_error": float(0.10 * feature[1]),
                "bearing_error": float(180.0 * feature[2]),
                "features": feature.astype(float).tolist(),
                "ppo_logit": float(logits[index]),
                "ppo_probability": float(probabilities[index]),
                "ppo_rank": int(_rank_descending(probabilities)[index]),
                "selected": index == selected,
            })
        max_delta = 0.0
        all_match = True
        for _ in range(max(0, args.permutations - 1)):
            permutation = rng.permutation(len(features))
            _, permuted_probabilities, permuted_selected = _policy_outputs(model, features[permutation])
            restored = np.empty_like(permuted_probabilities)
            restored[permutation] = permuted_probabilities
            max_delta = max(max_delta, float(np.max(np.abs(restored - probabilities))))
            restored_selected = int(permutation[permuted_selected])
            all_match = all_match and restored_selected == selected
        permutation_deltas.append(max_delta)
        permutation_matches.append(all_match)
        records.append({
            "record_type": "real",
            "filename": scenario["filename"],
            "scenario_id": scenario["group_id"],
            "generation": _generation(scenario["filename"]),
            "split": split_map[scenario["group_id"]],
            "front_size": len(routes),
            "selected_index": selected,
            "selected_source_idx": int(routes[selected]["source_idx"]),
            "routes": routes,
            "permutation_max_probability_delta": max_delta,
            "permutation_choice_invariant": all_match,
        })
    _jsonl_write(output_dir / "inference_dataset.jsonl", records)
    report = {
        "num_fronts": len(records),
        "num_routes": sum(len(record["routes"]) for record in records),
        "split_fronts": {
            split: sum(record["split"] == split for record in records)
            for split in ("fit", "calibration", "audit")
        },
        "permutation_choice_accuracy": float(np.mean(permutation_matches)),
        "permutation_max_probability_delta": float(max(permutation_deltas, default=0.0)),
    }
    _json_dump(output_dir / "extraction_report.json", report)
    print(
        f"Inference PPO: {report['num_fronts']} fronti, {report['num_routes']} rotte | "
        f"invarianza permutazioni={100.0 * report['permutation_choice_accuracy']:.2f}%"
    )
    return report


def _non_dominated_indices(features: np.ndarray) -> list[int]:
    """Indices of minimisation-objective non-dominated routes."""
    result = []
    for index, candidate in enumerate(features):
        dominated = False
        for other_index, other in enumerate(features):
            if other_index == index:
                continue
            if np.all(other <= candidate + 1e-9) and np.any(other < candidate - 1e-9):
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result


def generate_probing_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    real_records = _jsonl_read(output_dir / "inference_dataset.jsonl")
    anchors = [record for record in real_records if record["split"] == "fit" and record["front_size"] >= 1]
    if not anchors:
        raise ValueError("Nessun fronte nello split fit per il probing")
    model = MaskablePPO.load(args.model, device=args.device)
    rng = random.Random(args.seed)
    probe_records: list[dict[str, Any]] = []
    cell_summary: dict[str, dict[str, Any]] = {}
    for rule_index, rule in enumerate(RULES):
        probe_feature = _representative_feature(rule)
        # Round-robin over generations avoids filling all contexts from the
        # first (usually generation 50) bucket.
        generation_buckets: dict[int, list[dict[str, Any]]] = {}
        for anchor in anchors:
            generation_buckets.setdefault(int(anchor["generation"]), []).append(anchor)
        for bucket in generation_buckets.values():
            rng.shuffle(bucket)
        candidates = []
        max_bucket = max(map(len, generation_buckets.values()))
        for position in range(max_bucket):
            for generation in sorted(generation_buckets):
                bucket = generation_buckets[generation]
                if position < len(bucket):
                    candidates.append(bucket[position])
        accepted = 0
        attempts = 0
        ranks: list[int] = []
        probabilities: list[float] = []
        selections: list[bool] = []
        for anchor in candidates:
            if accepted >= args.probe_contexts:
                break
            attempts += 1
            original = np.asarray([route["features"] for route in anchor["routes"]], dtype=np.float32)
            combined = np.vstack((original, probe_feature))
            keep = _non_dominated_indices(combined)
            probe_original_index = len(combined) - 1
            if probe_original_index not in keep:
                continue
            pruned = combined[keep]
            if len(pruned) < 2 or len(pruned) > int(model.action_space.n):
                continue
            probe_index = keep.index(probe_original_index)
            logits, policy_probabilities, selected = _policy_outputs(model, pruned)
            ranking = _rank_descending(policy_probabilities)
            routes = []
            for index, feature in enumerate(pruned):
                source_index = keep[index]
                routes.append({
                    "source_idx": -1 if source_index == probe_original_index else int(
                        anchor["routes"][source_index]["source_idx"]
                    ),
                    "features": feature.astype(float).tolist(),
                    "ppo_logit": float(logits[index]),
                    "ppo_probability": float(policy_probabilities[index]),
                    "ppo_rank": int(ranking[index]),
                    "selected": index == selected,
                    "is_probe": index == probe_index,
                })
            ranks.append(int(ranking[probe_index]))
            probabilities.append(float(policy_probabilities[probe_index]))
            selections.append(selected == probe_index)
            probe_records.append({
                "record_type": "counterfactual",
                "filename": f"probe_{rule_index:02d}_{accepted:03d}",
                "scenario_id": anchor["scenario_id"],
                "generation": anchor["generation"],
                "split": "probe",
                "front_size": len(routes),
                "selected_index": selected,
                "probe_index": probe_index,
                "probe_rule_index": rule_index,
                "probe_rule": list(rule),
                "anchor_filename": anchor["filename"],
                "routes": routes,
            })
            accepted += 1
        key = "/".join(rule)
        cell_summary[key] = {
            "rule_index": rule_index,
            "accepted_contexts": accepted,
            "attempted_contexts": attempts,
            "selection_rate": float(np.mean(selections)) if selections else None,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "mean_probability": float(np.mean(probabilities)) if probabilities else None,
            "coverage": "counterfactual" if accepted else "unsupported",
        }
    _jsonl_write(output_dir / "probing_dataset.jsonl", probe_records)
    report = {
        "num_probe_fronts": len(probe_records),
        "target_contexts_per_rule": args.probe_contexts,
        "cells": cell_summary,
    }
    _json_dump(output_dir / "probing_report.json", report)
    print(
        f"Probing: {len(probe_records)} fronti controfattuali, "
        f"{sum(value['accepted_contexts'] > 0 for value in cell_summary.values())}/27 celle coperte"
    )
    return report


class ConsequentModel(nn.Module):
    """Identifiable zero-order fuzzy model with fixed antecedents.

    Only one scalar consequent is learned for each rule.  The linguistic
    weights are derived deterministically afterwards; learning five free
    weights per rule would be non-identifiable because many distributions
    share the same centroid.
    """

    def __init__(self, initial_values: np.ndarray, centroids: np.ndarray) -> None:
        super().__init__()
        self.raw_values = nn.Parameter(0.05 * th.randn(len(RULES), dtype=th.float32))
        self.register_buffer("centroids", th.as_tensor(centroids, dtype=th.float32))
        self.register_buffer("prior_values", th.as_tensor(initial_values, dtype=th.float32))

    def rule_values(self) -> th.Tensor:
        minimum, maximum = self.centroids[0], self.centroids[-1]
        return minimum + (maximum - minimum) * th.sigmoid(self.raw_values)

    def forward(self, activations: th.Tensor) -> th.Tensor:
        numerator = activations @ self.rule_values()
        denominator = activations.sum(dim=-1).clamp_min(EPSILON)
        return numerator / denominator


def _weights_from_rule_values(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Unique piecewise-linear linguistic projection of crisp consequents."""
    values = np.asarray(values, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    result = np.zeros((len(values), len(centroids)), dtype=np.float64)
    for row, value in enumerate(values):
        if value <= centroids[0]:
            result[row, 0] = 1.0
            continue
        if value >= centroids[-1]:
            result[row, -1] = 1.0
            continue
        upper = int(np.searchsorted(centroids, value, side="right"))
        lower = upper - 1
        fraction = float((value - centroids[lower]) / (centroids[upper] - centroids[lower]))
        result[row, lower] = 1.0 - fraction
        result[row, upper] = fraction
    return result.astype(np.float32)


def _pack_records(
    records: Sequence[dict[str, Any]], basis: FuzzyBasis, capacity: int, probe_weight: float = 0.25
) -> dict[str, th.Tensor]:
    count = len(records)
    activation = np.zeros((count, capacity, len(RULES)), dtype=np.float32)
    target = np.zeros((count, capacity), dtype=np.float32)
    mask = np.zeros((count, capacity), dtype=bool)
    weights = np.ones(count, dtype=np.float32)
    for row_index, record in enumerate(records):
        features = np.asarray([route["features"] for route in record["routes"]], dtype=np.float32)
        size = len(features)
        if size > capacity:
            raise ValueError(f"Record {record['filename']} oltre capacita': {size}>{capacity}")
        activation[row_index, :size] = basis.activation(features)
        target[row_index, :size] = [route["ppo_probability"] for route in record["routes"]]
        mask[row_index, :size] = True
        if record["record_type"] == "counterfactual":
            weights[row_index] = probe_weight
    return {
        "activation": th.from_numpy(activation),
        "target": th.from_numpy(target),
        "mask": th.from_numpy(mask),
        "weight": th.from_numpy(weights),
    }


def _surrogate_distribution(scores: th.Tensor, mask: th.Tensor, temperature: float) -> th.Tensor:
    masked_scores = (scores / temperature).masked_fill(~mask, -1e9)
    return th.softmax(masked_scores, dim=-1)


def _metrics_from_tensors(
    model: ConsequentModel,
    tensors: dict[str, th.Tensor],
    temperature: float,
) -> dict[str, float]:
    with th.no_grad():
        scores = model(tensors["activation"])
        distribution = _surrogate_distribution(scores, tensors["mask"], temperature)
        predicted = distribution.argmax(dim=-1)
        expected = tensors["target"].argmax(dim=-1)
        fidelity = (predicted == expected).float()
        top2 = distribution.topk(k=min(2, distribution.shape[1]), dim=-1).indices
        top2_match = (top2 == expected.unsqueeze(-1)).any(dim=-1).float()
        row = th.arange(len(predicted))
        probability_regret = tensors["target"][row, expected] - tensors["target"][row, predicted]
    return {
        "num_fronts": int(len(predicted)),
        "top1_fidelity": float(fidelity.mean().item()) if len(fidelity) else 0.0,
        "top2_fidelity": float(top2_match.mean().item()) if len(top2_match) else 0.0,
        "mean_ppo_probability_regret": float(probability_regret.mean().item()) if len(predicted) else 0.0,
        "max_ppo_probability_regret": float(probability_regret.max().item()) if len(predicted) else 0.0,
    }


def fit_surrogate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    real = _jsonl_read(output_dir / "inference_dataset.jsonl")
    probe_path = output_dir / "probing_dataset.jsonl"
    probes = _jsonl_read(probe_path) if probe_path.exists() else []
    fit_records = [record for record in real if record["split"] == "fit"] + probes
    calibration_records = [record for record in real if record["split"] == "calibration"]
    if not fit_records or not calibration_records:
        raise ValueError("Servono record fit e calibration prima del fitting")
    basis = FuzzyBasis(FuzzyDecisionMaker())
    initial_labels, initial_values = _initial_rule_labels(basis)
    manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    capacity = int(manifest["capacity"])
    train_tensors = _pack_records(fit_records, basis, capacity, args.probe_weight)
    calibration_tensors = _pack_records(calibration_records, basis, capacity)
    best_state: dict[str, th.Tensor] | None = None
    best_key = (-math.inf, -math.inf)
    best_epoch = 0
    best_restart = 0
    all_histories: list[dict[str, Any]] = []
    model: ConsequentModel | None = None
    for restart in range(args.restarts):
        th.manual_seed(args.seed + restart)
        candidate = ConsequentModel(initial_values, basis.output_centroids)
        optimizer = th.optim.Adam(candidate.parameters(), lr=args.learning_rate)
        patience = 0
        history: list[dict[str, float]] = []
        local_best_key = (-math.inf, -math.inf)
        for epoch in range(1, args.epochs + 1):
            scores = candidate(train_tensors["activation"])
            predicted_distribution = _surrogate_distribution(scores, train_tensors["mask"], args.temperature)
            target = train_tensors["target"].clamp_min(EPSILON)
            kl_per_route = target * (target.log() - predicted_distribution.clamp_min(EPSILON).log())
            kl = (kl_per_route.sum(dim=-1) * train_tensors["weight"]).mean()
            expected_index = target.argmax(dim=-1)
            row = th.arange(len(scores))
            chosen_score = scores[row, expected_index].unsqueeze(-1)
            competitor_margin = (args.margin - (chosen_score - scores)).clamp_min(0.0)
            competitor_mask = train_tensors["mask"].clone()
            competitor_mask[row, expected_index] = False
            ranking_per_front = (
                (competitor_margin * competitor_mask).sum(dim=-1)
                / competitor_mask.sum(dim=-1).clamp_min(1)
            )
            ranking = (ranking_per_front * train_tensors["weight"]).mean()
            scale = (candidate.centroids[-1] - candidate.centroids[0]).clamp_min(EPSILON)
            prior = th.mean(((candidate.rule_values() - candidate.prior_values) / scale) ** 2)
            loss = kl + args.ranking_weight * ranking + args.prior_weight * prior
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
                calibration = _metrics_from_tensors(candidate, calibration_tensors, args.temperature)
                current_key = (
                    calibration["top1_fidelity"],
                    -calibration["mean_ppo_probability_regret"],
                )
                history.append({
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    **calibration,
                })
                if current_key > best_key:
                    best_key = current_key
                    best_state = {
                        key: value.detach().clone() for key, value in candidate.state_dict().items()
                    }
                    best_epoch = epoch
                    best_restart = restart
                if current_key > local_best_key:
                    local_best_key = current_key
                    patience = 0
                else:
                    patience += args.eval_every
                if patience >= args.patience:
                    break
        all_histories.append({"restart": restart, "seed": args.seed + restart, "history": history})
        model = candidate
    if best_state is None:
        raise RuntimeError("Il fitting non ha prodotto un modello valido")
    assert model is not None
    model.load_state_dict(best_state)
    rule_values = model.rule_values().detach().cpu().numpy()
    weights = _weights_from_rule_values(rule_values, basis.output_centroids)
    real_fit_routes = [route for record in real if record["split"] == "fit" for route in record["routes"]]
    probe_routes = [route for record in probes for route in record["routes"] if route.get("is_probe")]
    real_activation = basis.activation(np.asarray([route["features"] for route in real_fit_routes]))
    probe_activation = (
        basis.activation(np.asarray([route["features"] for route in probe_routes]))
        if probe_routes else np.zeros((0, len(RULES)), dtype=np.float32)
    )
    real_support = real_activation.sum(axis=0)
    probe_support = probe_activation.sum(axis=0)
    # Estimate how context-dependent the PPO preference is for each fuzzy
    # region.  Rank percentiles are comparable across variable front sizes.
    desirability = []
    for record in real:
        if record["split"] != "fit":
            continue
        size = len(record["routes"])
        for route in record["routes"]:
            rank = int(route["ppo_rank"])
            desirability.append(1.0 if size == 1 else 1.0 - (rank - 1) / (size - 1))
    desirability_array = np.asarray(desirability, dtype=np.float32)
    weighted_mean = (
        (real_activation * desirability_array[:, None]).sum(axis=0)
        / np.maximum(real_support, EPSILON)
    )
    context_variance = (
        (real_activation * (desirability_array[:, None] - weighted_mean[None, :]) ** 2).sum(axis=0)
        / np.maximum(real_support, EPSILON)
    )
    # A cell unseen both in real data and probing is not learnable.  Restore
    # its initial consequent instead of exporting an optimizer artefact.
    unsupported = (real_support < args.min_real_support) & (probe_support <= EPSILON)
    if np.any(unsupported):
        rule_values[unsupported] = initial_values[unsupported]
        weights = _weights_from_rule_values(rule_values, basis.output_centroids)
    rules = []
    for index, rule in enumerate(RULES):
        if real_support[index] >= args.min_real_support:
            coverage = "observed"
        elif probe_support[index] > 0:
            coverage = "counterfactual"
        else:
            coverage = "unsupported"
        dominant_index = int(np.argmax(weights[index]))
        rules.append({
            "index": index,
            "antecedents": {
                "path_ratio": rule[0],
                "horizontal_error": rule[1],
                "bearing_error": rule[2],
            },
            "consequent_weights": {
                name: float(weights[index, term_index])
                for term_index, name in enumerate(OUTPUT_TERMS)
            },
            "continuous_consequent": float(rule_values[index]),
            "primary_consequent": OUTPUT_TERMS[dominant_index],
            "confidence": float(weights[index, dominant_index]),
            "real_support": float(real_support[index]),
            "counterfactual_support": float(probe_support[index]),
            "context_variance": float(context_variance[index]),
            "coverage": coverage,
            "initial_consequent": initial_labels[index],
            "initial_value": float(initial_values[index]),
        })
    artifact = {
        "format_version": 1,
        "type": "weighted_zero_order_fuzzy_surrogate",
        "source_model": manifest["model"],
        "source_model_sha256": manifest["model_sha256"],
        "antecedents": {
            "path_ratio": list(PATH_TERMS),
            "horizontal_error": list(HORIZONTAL_TERMS),
            "bearing_error": list(BEARING_TERMS),
            "membership_source": "recommender.rlfuzzy.FuzzyDecisionMaker",
            "operator": "product",
        },
        "output_terms": list(OUTPUT_TERMS),
        "output_centroids": {
            name: float(value) for name, value in zip(OUTPUT_TERMS, basis.output_centroids)
        },
        "temperature": args.temperature,
        "rules": rules,
        "training": {
            "seed": args.seed,
            "restarts": args.restarts,
            "best_restart": best_restart,
            "epochs_completed": all_histories[best_restart]["history"][-1]["epoch"],
            "best_epoch": best_epoch,
            "learning_rate": args.learning_rate,
            "ranking_weight": args.ranking_weight,
            "prior_weight": args.prior_weight,
            "probe_weight": args.probe_weight,
            "histories": all_histories,
        },
    }
    _json_dump(output_dir / "surrogate_fuzzy.json", artifact)
    th.save({"artifact": artifact}, output_dir / "surrogate_fuzzy.pt")
    print(
        f"Surrogate distillato: restart {best_restart}, best epoch {best_epoch} | "
        f"calibration fidelity={100.0 * best_key[0]:.2f}%"
    )
    return artifact


class WeightedSurrogateFIS:
    """Portable inference interface backed by ``surrogate_fuzzy.json``."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.basis = FuzzyBasis(FuzzyDecisionMaker())
        self.rule_values = np.asarray(
            [rule["continuous_consequent"] for rule in artifact["rules"]], dtype=np.float32
        )

    @classmethod
    def load(cls, path: str | Path) -> "WeightedSurrogateFIS":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def score_features(self, features: np.ndarray) -> np.ndarray:
        activation = self.basis.activation(features)
        denominator = np.maximum(activation.sum(axis=-1), EPSILON)
        return (activation @ self.rule_values) / denominator

    def select_features(self, features: np.ndarray) -> tuple[int, np.ndarray]:
        scores = np.asarray(self.score_features(features), dtype=np.float32)
        return int(np.argmax(scores)), scores

    def evaluate_solution(
        self, path_length: float, horizontal_error: float, bearing_error: float, direct_distance: float
    ) -> float:
        feature = self.basis.fuzzy.policy_features(
            path_length, horizontal_error, bearing_error, direct_distance
        )
        return float(self.score_features(feature))


class LinguisticSurrogateFIS(WeightedSurrogateFIS):
    """Single-consequent linguistic projection of the weighted surrogate."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        super().__init__(artifact)
        self.primary_terms = [rule["primary_consequent"] for rule in artifact["rules"]]

    def score_features(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        single = array.ndim == 1
        flat = array.reshape(-1, len(ROUTE_FEATURES))
        activations = self.basis.activation(flat, operator="min")
        universe = self.basis.fuzzy.preference.universe
        output_memberships = {
            name: np.asarray(self.basis.fuzzy.preference[name].mf, dtype=np.float64)
            for name in OUTPUT_TERMS
        }
        scores = []
        for route_activation in activations:
            aggregate = np.zeros_like(universe, dtype=np.float64)
            for rule_index, strength in enumerate(route_activation):
                consequent = output_memberships[self.primary_terms[rule_index]]
                aggregate = np.maximum(aggregate, np.minimum(float(strength), consequent))
            if np.max(aggregate) <= EPSILON:
                scores.append(50.0)
            else:
                scores.append(float(fuzz.defuzz(universe, aggregate, "centroid")))
        result = np.asarray(scores, dtype=np.float32)
        return result[0] if single else result


class WeightedMamdaniSurrogateFIS(WeightedSurrogateFIS):
    """Mamdani compilation retaining both adjacent linguistic consequents."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        super().__init__(artifact)
        self.consequent_weights = np.asarray([
            [rule["consequent_weights"][name] for name in OUTPUT_TERMS]
            for rule in artifact["rules"]
        ], dtype=np.float32)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        single = array.ndim == 1
        flat = array.reshape(-1, len(ROUTE_FEATURES))
        activations = self.basis.activation(flat, operator="min")
        universe = self.basis.fuzzy.preference.universe
        output_memberships = np.stack([
            np.asarray(self.basis.fuzzy.preference[name].mf, dtype=np.float64)
            for name in OUTPUT_TERMS
        ])
        scores = []
        for route_activation in activations:
            aggregate = np.zeros_like(universe, dtype=np.float64)
            for rule_index, strength in enumerate(route_activation):
                weighted_strengths = float(strength) * self.consequent_weights[rule_index]
                for term_index, weighted_strength in enumerate(weighted_strengths):
                    if weighted_strength <= EPSILON:
                        continue
                    aggregate = np.maximum(
                        aggregate,
                        np.minimum(float(weighted_strength), output_memberships[term_index]),
                    )
            scores.append(
                50.0 if np.max(aggregate) <= EPSILON
                else float(fuzz.defuzz(universe, aggregate, "centroid"))
            )
        result = np.asarray(scores, dtype=np.float32)
        return result[0] if single else result


def _evaluate_records(
    surrogate: WeightedSurrogateFIS, records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    top1: list[float] = []
    top2: list[float] = []
    regrets: list[float] = []
    rank_correlations: list[float] = []
    by_generation: dict[int, list[float]] = {}
    details = []
    for record in records:
        features = np.asarray([route["features"] for route in record["routes"]], dtype=np.float32)
        scores = surrogate.score_features(features)
        predicted = int(np.argmax(scores))
        probabilities = np.asarray([route["ppo_probability"] for route in record["routes"]])
        expected = int(np.argmax(probabilities))
        order = np.argsort(-scores, kind="stable")
        match = predicted == expected
        top1.append(float(match))
        top2.append(float(expected in order[:min(2, len(order))]))
        regret = float(probabilities[expected] - probabilities[predicted])
        regrets.append(regret)
        if len(scores) > 1 and np.std(scores) > EPSILON and np.std(probabilities) > EPSILON:
            score_ranks = _rank_descending(scores).astype(float)
            policy_ranks = _rank_descending(probabilities).astype(float)
            correlation = float(np.corrcoef(score_ranks, policy_ranks)[0, 1])
            if np.isfinite(correlation):
                rank_correlations.append(correlation)
        generation = int(record["generation"])
        by_generation.setdefault(generation, []).append(float(match))
        details.append({
            "filename": record["filename"],
            "split": record["split"],
            "expected_index": expected,
            "predicted_index": predicted,
            "match": match,
            "ppo_probability_regret": regret,
        })
    return {
        "num_fronts": len(records),
        "top1_fidelity": float(np.mean(top1)) if top1 else 0.0,
        "top2_fidelity": float(np.mean(top2)) if top2 else 0.0,
        "mean_rank_correlation": float(np.mean(rank_correlations)) if rank_correlations else 0.0,
        "mean_ppo_probability_regret": float(np.mean(regrets)) if regrets else 0.0,
        "max_ppo_probability_regret": float(np.max(regrets)) if regrets else 0.0,
        "by_generation": {
            str(generation): {
                "num_fronts": len(values),
                "top1_fidelity": float(np.mean(values)),
            }
            for generation, values in sorted(by_generation.items())
        },
        "details": details,
    }


def _evaluate_teacher_chain(
    fuzzy: FuzzyDecisionMaker,
    surrogate: WeightedSurrogateFIS,
    records: Sequence[dict[str, Any]],
) -> dict[str, float]:
    exact_teacher_ppo = []
    reward_equivalent_teacher_ppo = []
    surrogate_teacher = []
    for record in records:
        features = np.asarray([route["features"] for route in record["routes"]], dtype=np.float32)
        probabilities = np.asarray([route["ppo_probability"] for route in record["routes"]])
        ppo_index = int(np.argmax(probabilities))
        teacher_scores = np.asarray([
            fuzzy.evaluate_solution(
                1.0 + 1.5 * float(feature[0]),
                0.10 * float(feature[1]),
                180.0 * float(feature[2]),
                1.0,
            )
            for feature in features
        ])
        teacher_index = int(np.argmax(teacher_scores))
        surrogate_index = int(np.argmax(surrogate.score_features(features)))
        exact_teacher_ppo.append(teacher_index == ppo_index)
        teacher_regret = float(np.max(teacher_scores) - teacher_scores[ppo_index])
        reward_equivalent_teacher_ppo.append(teacher_regret <= MATCH_TOLERANCE)
        surrogate_teacher.append(surrogate_index == teacher_index)
    return {
        "num_fronts": len(records),
        "teacher_ppo_exact_fidelity": float(np.mean(exact_teacher_ppo)) if records else 0.0,
        "teacher_ppo_reward_equivalence": (
            float(np.mean(reward_equivalent_teacher_ppo)) if records else 0.0
        ),
        "surrogate_teacher_exact_agreement": (
            float(np.mean(surrogate_teacher)) if records else 0.0
        ),
    }


def _write_rule_table(output_dir: Path, artifact: dict[str, Any]) -> None:
    with (output_dir / "rule_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "path_ratio", "horizontal_error", "bearing_error",
            "initial_consequent", "primary_consequent", "continuous_consequent",
            "confidence", "real_support", "counterfactual_support", "coverage",
            "context_variance",
        ])
        writer.writeheader()
        for rule in artifact["rules"]:
            row = {
                "index": rule["index"],
                **rule["antecedents"],
                **{key: rule.get(key) for key in (
                    "initial_consequent", "primary_consequent", "continuous_consequent",
                    "confidence", "real_support", "counterfactual_support", "coverage",
                    "context_variance",
                )},
            }
            writer.writerow(row)


def _write_plots(output_dir: Path, artifact: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    values = np.asarray([rule["continuous_consequent"] for rule in artifact["rules"]]).reshape(3, 3, 3)
    support = np.asarray([rule["real_support"] for rule in artifact["rules"]]).reshape(3, 3, 3)
    initial = np.asarray([rule["initial_value"] for rule in artifact["rules"]])
    learned = values.reshape(-1)
    for name, cube, colour_map in (("consequents", values, "viridis"), ("support", support, "magma")):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for index, path_name in enumerate(PATH_TERMS):
            image = axes[index].imshow(cube[index], origin="lower", cmap=colour_map, aspect="auto")
            axes[index].set_title(path_name)
            axes[index].set_xticks(range(3), BEARING_TERMS, rotation=25)
            axes[index].set_yticks(range(3), HORIZONTAL_TERMS)
            axes[index].set_xlabel("bearing")
            axes[index].set_ylabel("horizontal")
            figure.colorbar(image, ax=axes[index], shrink=0.8)
        figure.savefig(plots / f"rule_{name}.png", dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    indices = np.arange(len(RULES))
    axis.plot(indices, initial, marker="o", label="Fuzzy iniziale")
    axis.plot(indices, learned, marker="o", label="Surrogate PPO")
    axis.set_xlabel("Indice regola")
    axis.set_ylabel("Conseguente 0-100")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(plots / "initial_vs_surrogate.png", dpi=160)
    plt.close(figure)


def evaluate_surrogate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    artifact = json.loads((output_dir / "surrogate_fuzzy.json").read_text(encoding="utf-8"))
    records = _jsonl_read(output_dir / "inference_dataset.jsonl")
    weighted = WeightedSurrogateFIS(artifact)
    mamdani_weighted = WeightedMamdaniSurrogateFIS(artifact)
    linguistic = LinguisticSurrogateFIS(artifact)
    teacher = FuzzyDecisionMaker()
    report: dict[str, Any] = {
        "weighted": {}, "mamdani_weighted": {}, "linguistic": {}, "teacher_chain": {}
    }
    for split in ("fit", "calibration", "audit"):
        split_records = [record for record in records if record["split"] == split]
        report["weighted"][split] = _evaluate_records(weighted, split_records)
        report["mamdani_weighted"][split] = _evaluate_records(mamdani_weighted, split_records)
        report["linguistic"][split] = _evaluate_records(linguistic, split_records)
        report["teacher_chain"][split] = _evaluate_teacher_chain(
            teacher, weighted, split_records
        )
    audit_weighted = report["weighted"]["audit"]
    audit_mamdani_weighted = report["mamdani_weighted"]["audit"]
    audit_linguistic = report["linguistic"]["audit"]
    extraction_report_path = output_dir / "extraction_report.json"
    extraction_report = (
        json.loads(extraction_report_path.read_text(encoding="utf-8"))
        if extraction_report_path.exists() else {}
    )
    report["acceptance"] = {
        "weighted_top1_at_least_90": audit_weighted["top1_fidelity"] >= 0.90,
        "weighted_top2_at_least_97": audit_weighted["top2_fidelity"] >= 0.97,
        "weighted_mamdani_drop_at_most_3pp": (
            audit_weighted["top1_fidelity"] - audit_mamdani_weighted["top1_fidelity"] <= 0.03
        ),
        "permutation_choice_invariant": (
            extraction_report.get("permutation_choice_accuracy") == 1.0
        ),
    }
    report["accepted"] = all(report["acceptance"].values())
    _json_dump(output_dir / "fidelity_report.json", report)
    _write_rule_table(output_dir, artifact)
    _write_plots(output_dir, artifact)
    print("=" * 72)
    print(f"SURROGATE PESATO audit top-1: {100.0 * audit_weighted['top1_fidelity']:.2f}%")
    print(f"SURROGATE PESATO audit top-2: {100.0 * audit_weighted['top2_fidelity']:.2f}%")
    print(
        "MAMDANI PESATO audit top-1: "
        f"{100.0 * audit_mamdani_weighted['top1_fidelity']:.2f}%"
    )
    print(f"MAMDANI LINGUISTICO audit top-1: {100.0 * audit_linguistic['top1_fidelity']:.2f}%")
    print(
        "TEACHER fuzzy -> PPO audit: "
        f"{100.0 * report['teacher_chain']['audit']['teacher_ppo_reward_equivalence']:.2f}%"
    )
    print(
        "SURROGATE -> teacher audit: "
        f"{100.0 * report['teacher_chain']['audit']['surrogate_teacher_exact_agreement']:.2f}%"
    )
    print(f"Accettato: {'SI' if report['accepted'] else 'NO'}")
    print("=" * 72)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("extract", "probe", "fit", "evaluate", "all"), nargs="?", default="all"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-folder", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--probe-contexts", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--probe-weight", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--min-real-support", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in ("extract", "all"):
        extract_policy_dataset(args)
    if args.command in ("probe", "all"):
        generate_probing_dataset(args)
    if args.command in ("fit", "all"):
        fit_surrogate(args)
    if args.command in ("evaluate", "all"):
        evaluate_surrogate(args)


if __name__ == "__main__":
    main()
