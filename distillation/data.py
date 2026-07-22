from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from recommender.policy_contract import MAX_PARETO_SIZE, PADDING_VALUE, ROUTE_FEATURES


@dataclass(frozen=True)
class Front:
    filename: str
    group_id: str
    features: np.ndarray


def _metric(solution: dict[str, Any], name: str, index: int) -> float:
    value = solution.get(name)
    if value is None:
        value = (solution.get("metriche_reali") or {}).get(name)
    if value is None:
        fitness = solution.get("fitness") or []
        value = fitness[index] if len(fitness) > index else None
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Metrica {name} mancante o non finita")
    return float(value)


def normalized_features(document: dict[str, Any]) -> np.ndarray:
    scenario = document.get("scenario") or {}
    aircraft = scenario.get("ac0") or {}
    start, target = aircraft.get("p0"), aircraft.get("p_target")
    if not start or not target or len(start) < 2 or len(target) < 2:
        raise ValueError("scenario.ac0.p0/p_target mancanti")
    direct = math.hypot(float(target[0]) - float(start[0]), float(target[1]) - float(start[1]))
    if direct <= 0.0:
        raise ValueError("Distanza diretta nulla")
    rows = []
    for solution in document.get("population") or []:
        real = solution.get("metriche_reali") or {}
        if solution.get("collisione") or solution.get("conflict") or real.get("collisione") or real.get("conflict"):
            rows.append((1.0, 1.0, 1.0))
            continue
        path = _metric(solution, "path_length", 0)
        horizontal = _metric(solution, "horizontal_target_error_nm", 1)
        bearing = _metric(solution, "final_bearing_error_deg", 2)
        rows.append((
            (np.clip(path / direct, 1.0, 2.5) - 1.0) / 1.5,
            np.clip(horizontal, 0.0, 0.10) / 0.10,
            np.clip(abs(bearing), 0.0, 180.0) / 180.0,
        ))
    if not rows:
        raise ValueError("Fronte Pareto vuoto")
    if len(rows) > MAX_PARETO_SIZE:
        raise ValueError(f"Fronte da {len(rows)} rotte, massimo {MAX_PARETO_SIZE}")
    return np.asarray(rows, dtype=np.float32)


def load_fronts(folder: str | Path) -> list[Front]:
    paths = sorted(Path(folder).glob("*_pareto_front_*_pruned.json"))
    if not paths:
        raise ValueError(f"Nessun fronte Pareto in {folder}")
    fronts = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            match = re.match(r"((?:exp_)?\d+)_pareto_front_", path.name)
            fronts.append(Front(path.name, match.group(1) if match else path.name, normalized_features(document)))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"JSON Pareto non valido {path}: {exc}") from exc
    return fronts


def split_train_validation(
    fronts: Sequence[Front], contract: dict[str, Any], seed: int = 42
) -> tuple[list[Front], list[Front], list[str]]:
    groups = sorted({front.group_id for front in fronts})
    if len(groups) < 2:
        raise ValueError("Servono almeno due scenari per train e validation")
    validation = set(map(str, contract.get("validation_groups") or ())) & set(groups)
    if not validation:
        shuffled = np.random.default_rng(seed).permutation(groups)
        validation = set(shuffled[:max(1, round(0.2 * len(groups)))])
    train = [front for front in fronts if front.group_id not in validation]
    valid = [front for front in fronts if front.group_id in validation]
    if not train or not valid:
        raise ValueError("Split train/validation vuoto")
    return train, valid, sorted(validation)


def assert_disjoint(first: Sequence[Front], second: Sequence[Front]) -> None:
    overlap = sorted({front.group_id for front in first} & {front.group_id for front in second})
    if overlap:
        raise ValueError(f"Leakage tra scenari: {', '.join(overlap[:10])}")


def pad_fronts(fronts: Sequence[Front], capacity: int) -> tuple[np.ndarray, np.ndarray]:
    observations = np.full((len(fronts), capacity, len(ROUTE_FEATURES)), PADDING_VALUE, dtype=np.float32)
    masks = np.zeros((len(fronts), capacity), dtype=bool)
    for row, front in enumerate(fronts):
        size = len(front.features)
        if size > capacity:
            raise ValueError(f"{front.filename}: {size} rotte oltre la capacita PPO {capacity}")
        observations[row, :size] = front.features
        masks[row, :size] = True
    return observations, masks


def _model_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    zipped = candidate.with_suffix(".zip")
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(f"Checkpoint PPO non trovato: {path}")


def load_policy(path: str | Path, device: str = "cpu") -> tuple[Any, Path, dict[str, Any]]:
    import torch
    from sb3_contrib import MaskablePPO

    checkpoint = _model_file(path)
    contract_path = checkpoint.parent / "model_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"model_contract.json mancante accanto a {checkpoint}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("problem") != "contextual_bandit" or not np.isclose(float(contract.get("gamma", 1.0)), 0.0):
        raise ValueError("Il checkpoint non dichiara un contextual bandit con gamma=0")
    if contract.get("action_shuffling") is not True:
        raise ValueError("Il checkpoint non dichiara action_shuffling=true")
    if tuple(contract.get("route_features") or ()) != ROUTE_FEATURES:
        raise ValueError("Feature del checkpoint incompatibili")
    if float(contract.get("padding", PADDING_VALUE)) != PADDING_VALUE:
        raise ValueError("Padding del checkpoint incompatibile")
    model = MaskablePPO.load(str(checkpoint), device=device)
    capacity = int(model.action_space.n)
    if capacity != int(contract.get("max_pareto_size", capacity)) or capacity > MAX_PARETO_SIZE:
        raise ValueError("Capacita PPO incompatibile con il contratto")
    if not isinstance(model.policy.action_net, torch.nn.Identity):
        raise ValueError("Il checkpoint non usa action_net=Identity")
    return model, checkpoint, contract


def ppo_logits(model: Any, fronts: Sequence[Front], batch_size: int = 256) -> list[np.ndarray]:
    import torch

    if not fronts:
        raise ValueError("Nessun fronte per l'estrazione dei logit")
    observations, _ = pad_fronts(fronts, int(model.action_space.n))
    flat = observations.reshape(len(fronts), -1)
    output = []
    for start in range(0, len(flat), batch_size):
        tensor = torch.as_tensor(flat[start:start + batch_size], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            logits = model.policy.mlp_extractor.forward_actor(tensor)
        output.append(logits.detach().cpu().numpy())
    matrix = np.concatenate(output)
    return [matrix[index, :len(front.features)].astype(np.float64) for index, front in enumerate(fronts)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
