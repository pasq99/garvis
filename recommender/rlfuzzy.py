"""Fuzzy-labelled contextual bandit for variable-size ATC Pareto fronts.

The action contract is intentionally simple: action ``i`` always refers to
route block ``i`` in the observation returned by the latest ``reset``.  Every
reset permutes those blocks and all aligned labels together.  Empty blocks are
padded with -1 and are made structurally unreachable through action masking.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
import glob
import json
import math
import os
import re
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch as th
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from recommender.policy_contract import MAX_PARETO_SIZE, PADDING_VALUE, ROUTE_FEATURES


MATCH_TOLERANCE = 0.5
ENCODER_WIDTH = 192


def pareto_json_files(folder: str, *, pruned_only: bool = True) -> list[str]:
    """Return canonical Pareto JSON files, excluding nested hold-out folders."""
    suffix = "*_pareto_front_*_pruned.json" if pruned_only else "*_pareto_front_*.json"
    return sorted(
        path for path in glob.glob(os.path.join(folder, suffix))
        if "_infeasible" not in os.path.basename(path)
    )


def dist_diretta_2d(p0: Sequence[float] | None, target: Sequence[float] | None) -> float | None:
    """Horizontal direct distance between start and target in scenario units."""
    if not p0 or not target or len(p0) < 2 or len(target) < 2:
        return None
    return math.hypot(float(target[0]) - float(p0[0]), float(target[1]) - float(p0[1]))


class FuzzyDecisionMaker:
    """Operational fuzzy oracle used to label GA Pareto solutions."""

    COLLISION_PENALTY_THRESHOLD = 10_000.0

    def __init__(self) -> None:
        # All inputs mirror the three objectives emitted by the current GA.
        # Path length is made comparable across scenarios by dividing it by the
        # straight horizontal distance from p0 to p_target.
        self.path_ratio = ctrl.Antecedent(np.arange(1.0, 2.501, 0.005), "path_ratio")
        self.path_ratio["direct"] = fuzz.trapmf(self.path_ratio.universe, [1.0, 1.0, 1.03, 1.10])
        self.path_ratio["moderate"] = fuzz.trimf(self.path_ratio.universe, [1.05, 1.15, 1.30])
        self.path_ratio["long"] = fuzz.trapmf(self.path_ratio.universe, [1.20, 1.40, 2.50, 2.50])

        target_tolerance = 0.10
        self.horizontal_err = ctrl.Antecedent(np.arange(0.0, target_tolerance + 0.0005, 0.0005), "horizontal_err")
        self.horizontal_err["precise"] = fuzz.trapmf(self.horizontal_err.universe, [0.0, 0.0, 0.005, 0.020])
        self.horizontal_err["acceptable"] = fuzz.trimf(self.horizontal_err.universe, [0.010, 0.040, 0.075])
        self.horizontal_err["limit"] = fuzz.trapmf(self.horizontal_err.universe, [0.055, 0.085, 0.10, 0.10])

        self.bearing_err = ctrl.Antecedent(np.arange(0.0, 180.1, 0.1), "bearing_err")
        self.bearing_err["aligned"] = fuzz.trapmf(self.bearing_err.universe, [0.0, 0.0, 1.0, 3.0])
        self.bearing_err["deviated"] = fuzz.trimf(self.bearing_err.universe, [2.0, 10.0, 30.0])
        self.bearing_err["poor"] = fuzz.trapmf(self.bearing_err.universe, [20.0, 45.0, 180.0, 180.0])

        self.preference = ctrl.Consequent(np.arange(0.0, 101.0, 1.0), "preference")
        self.preference["very_low"] = fuzz.trapmf(self.preference.universe, [0.0, 0.0, 10.0, 25.0])
        self.preference["low"] = fuzz.trimf(self.preference.universe, [15.0, 30.0, 45.0])
        self.preference["medium"] = fuzz.trimf(self.preference.universe, [35.0, 50.0, 65.0])
        self.preference["high"] = fuzz.trimf(self.preference.universe, [55.0, 72.0, 85.0])
        self.preference["optimal"] = fuzz.trapmf(self.preference.universe, [80.0, 90.0, 100.0, 100.0])

        # This is the original Mamdani teacher.  Keep it as an oracle for
        # evaluation/distillation; PPO itself never runs this control system.
        self.control_system = ctrl.ControlSystem([
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["precise"] & self.bearing_err["aligned"], self.preference["optimal"]),
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["precise"] & self.bearing_err["deviated"], self.preference["high"]),
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["precise"] & self.bearing_err["poor"], self.preference["medium"]),
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["acceptable"] & self.bearing_err["aligned"], self.preference["high"]),
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["acceptable"] & self.bearing_err["deviated"], self.preference["medium"]),
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["acceptable"] & self.bearing_err["poor"], self.preference["low"]),
            ctrl.Rule(self.path_ratio["direct"] & self.horizontal_err["limit"], self.preference["low"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["precise"] & self.bearing_err["aligned"], self.preference["high"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["precise"] & self.bearing_err["deviated"], self.preference["medium"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["precise"] & self.bearing_err["poor"], self.preference["low"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["acceptable"] & self.bearing_err["aligned"], self.preference["medium"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["acceptable"] & self.bearing_err["deviated"], self.preference["low"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["limit"], self.preference["very_low"]),
            ctrl.Rule(self.path_ratio["moderate"] & self.horizontal_err["acceptable"] & self.bearing_err["poor"], self.preference["very_low"]),
            ctrl.Rule(self.path_ratio["long"] & self.horizontal_err["precise"] & self.bearing_err["aligned"], self.preference["medium"]),
            ctrl.Rule(self.path_ratio["long"] & (self.horizontal_err["acceptable"] | self.horizontal_err["limit"] | self.bearing_err["deviated"] | self.bearing_err["poor"]), self.preference["very_low"]),
        ])

    def evaluate_solution(
        self,
        path_length: float,
        horizontal_err: float,
        bearing_err: float,
        dist_diretta: float | None = None,
    ) -> float:
        values = (path_length, horizontal_err, bearing_err)
        if not all(np.isfinite(values)):
            raise ValueError(f"Metriche non finite: {values}")
        if any(value >= self.COLLISION_PENALTY_THRESHOLD for value in values):
            return 0.0
        ratio = path_length / dist_diretta if dist_diretta and dist_diretta > 0.0 else 1.0
        simulation = ctrl.ControlSystemSimulation(self.control_system)
        simulation.input["path_ratio"] = float(np.clip(ratio, 1.0, 2.5))
        simulation.input["horizontal_err"] = float(np.clip(horizontal_err, 0.0, 0.10))
        simulation.input["bearing_err"] = float(np.clip(abs(bearing_err), 0.0, 180.0))
        try:
            simulation.compute()
            return float(simulation.output["preference"])
        except (KeyError, ValueError):
            return 0.0

    def _extract_solution_metrics(self, solution: dict[str, Any]) -> tuple[float, float, float]:
        """Read the canonical GA fitness: path, horizontal error, bearing error."""
        real = solution.get("metriche_reali") or {}
        if solution.get("collisione") or solution.get("conflict") or real.get("collisione") or real.get("conflict"):
            penalty = self.COLLISION_PENALTY_THRESHOLD
            return penalty, penalty, penalty
        fitness = solution.get("fitness") or []
        path = solution.get("path_length")
        h = solution.get("horizontal_target_error_nm")
        b = solution.get("final_bearing_error_deg")
        path = real.get("path_length") if path is None else path
        h = real.get("horizontal_target_error_nm") if h is None else h
        b = real.get("final_bearing_error_deg") if b is None else b
        path = fitness[0] if path is None and len(fitness) >= 1 else path
        h = fitness[1] if h is None and len(fitness) >= 2 else h
        b = fitness[2] if b is None and len(fitness) >= 3 else b
        named = {"path_length": path, "horizontal_target_error_nm": h,
                 "final_bearing_error_deg": b}
        missing = [name for name, value in named.items() if value is None]
        if missing:
            raise ValueError(f"Metriche mancanti: {', '.join(missing)}")
        result = tuple(float(value) for value in named.values())
        if not all(np.isfinite(result)):
            raise ValueError(f"Metriche non finite: {result}")
        return result  # type: ignore[return-value]

    def select_solution(
        self,
        pareto_solutions: Sequence[dict[str, Any]],
        epsilon: float = 0.0,
        dist_diretta: float | None = None,
    ) -> tuple[int, np.ndarray]:
        if not pareto_solutions:
            raise ValueError("Il fronte di Pareto e' vuoto")
        preferences = np.asarray([
            self.evaluate_solution(*self._extract_solution_metrics(solution), dist_diretta=dist_diretta)
            for solution in pareto_solutions
        ], dtype=np.float32)
        if epsilon > 0.0 and np.random.random() < epsilon:
            probabilities = preferences / preferences.sum() if preferences.sum() > 0 else None
            return int(np.random.choice(len(preferences), p=probabilities)), preferences
        return int(np.argmax(preferences)), preferences

    def policy_features(
        self,
        path_length: float,
        horizontal_err: float,
        bearing_err: float,
        direct_distance: float,
    ) -> np.ndarray:
        ratio = float(np.clip(path_length / direct_distance, 1.0, 2.5))
        return np.asarray([
            (ratio - 1.0) / 1.5,
            np.clip(horizontal_err, 0.0, 0.10) / 0.10,
            np.clip(abs(bearing_err), 0.0, 180.0) / 180.0,
        ], dtype=np.float32)


class ParetoSetExtractor(nn.Module):
    """Shared route scorer for exact permutation equivariance."""

    def __init__(self, max_routes: int, route_dim: int = len(ROUTE_FEATURES), hidden_dim: int = ENCODER_WIDTH) -> None:
        super().__init__()
        self.max_routes = max_routes
        self.route_dim = route_dim
        self.latent_dim_pi = max_routes
        self.latent_dim_vf = hidden_dim
        self.route_encoder = nn.Sequential(
            nn.Linear(route_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.route_scorer = nn.Linear(hidden_dim, 1)
        self.value_encoder = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())

    def _encode(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        routes = features.reshape(-1, self.max_routes, self.route_dim)
        valid = routes[..., 0] >= 0.0
        routes = th.where(valid.unsqueeze(-1), routes, th.zeros_like(routes))
        return self.route_encoder(routes), valid

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        encoded, _ = self._encode(features)
        return self.route_scorer(encoded).squeeze(-1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        encoded, valid = self._encode(features)
        weights = valid.unsqueeze(-1).to(encoded.dtype)
        mean_pool = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        max_pool = encoded.masked_fill(~valid.unsqueeze(-1), -th.inf).max(1).values
        max_pool = th.where(th.isfinite(max_pool), max_pool, th.zeros_like(max_pool))
        return self.value_encoder(th.cat((mean_pool, max_pool), dim=-1))

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class ParetoSetPolicy(MaskableActorCriticPolicy):
    """Maskable policy whose action head is a shared scorer, not an index MLP."""

    def _build_mlp_extractor(self) -> None:
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("ParetoSetPolicy richiede uno spazio Discrete")
        expected = self.action_space.n * len(ROUTE_FEATURES)
        if self.features_dim != expected:
            raise ValueError(f"Observation dim {self.features_dim}, attesa {expected}")
        self.mlp_extractor = ParetoSetExtractor(self.action_space.n)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        self.action_net = nn.Identity()  # actor latent already contains one logit per route
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            self.mlp_extractor.route_encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.mlp_extractor.value_encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.mlp_extractor.route_scorer.apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1.0))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


class MultiScenarioEnv(gym.Env):
    """One-step contextual bandit over full, variable-size Pareto fronts."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        input_folder: str | None,
        fuzzy_decider: FuzzyDecisionMaker,
        *,
        max_pareto_size: int | None = None,
        scenarios: Sequence[dict[str, Any]] | None = None,
        scenario_sampling: str = "random",
        permute_candidates: bool = True,
    ) -> None:
        super().__init__()
        if scenario_sampling not in {"random", "cycle"}:
            raise ValueError("scenario_sampling deve essere 'random' o 'cycle'")
        self.fuzzy_decider = fuzzy_decider
        self.scenario_sampling = scenario_sampling
        self.permute_candidates = bool(permute_candidates)
        self._cycle_idx = 0
        self.scenarios = list(scenarios) if scenarios is not None else self._load_scenarios(input_folder)
        if not self.scenarios:
            raise ValueError("Nessuno scenario valido caricato")
        observed_max = max(len(item["pareto_front"]) for item in self.scenarios)
        self.max_pareto_size = MAX_PARETO_SIZE if max_pareto_size is None else int(max_pareto_size)
        if self.max_pareto_size <= 0:
            raise ValueError("max_pareto_size deve essere positivo")
        if self.max_pareto_size > MAX_PARETO_SIZE:
            raise ValueError(f"max_pareto_size non puo' superare {MAX_PARETO_SIZE}")
        if self.max_pareto_size < observed_max:
            raise ValueError(
                f"Fronte da {observed_max} rotte > capacita' modello {self.max_pareto_size}; "
                "riaddestrare sul dataset completo"
            )
        obs_dim = self.max_pareto_size * len(ROUTE_FEATURES)
        self.observation_space = spaces.Box(PADDING_VALUE, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_pareto_size)
        self.current_scenario: dict[str, Any] | None = None
        self._action_permutation = np.empty(0, dtype=np.int64)
        sizes = [len(item["pareto_front"]) for item in self.scenarios]
        print(
            f"Caricati {len(sizes)} fronti (min={min(sizes)}, max={max(sizes)}); "
            f"observation/action capacity={self.max_pareto_size}."
        )

    def subset(self, scenarios: Sequence[dict[str, Any]], sampling: str) -> "MultiScenarioEnv":
        return MultiScenarioEnv(
            None, self.fuzzy_decider, max_pareto_size=self.max_pareto_size,
            scenarios=scenarios, scenario_sampling=sampling,
            permute_candidates=self.permute_candidates,
        )

    @staticmethod
    def scenario_group(filename: str) -> str:
        match = re.match(r"((?:exp_)?\d+)_pareto_front_", filename)
        return match.group(1) if match else filename

    def _load_scenarios(self, input_folder: str | None) -> list[dict[str, Any]]:
        if input_folder is None:
            raise ValueError("input_folder obbligatoria")
        paths = pareto_json_files(input_folder)
        if not paths:
            raise ValueError(f"Nessun JSON Pareto trovato in {input_folder}")
        print(f"Preparazione reward teacher fuzzy: {len(paths)} fronti...")
        result: list[dict[str, Any]] = []
        skipped = 0
        progress_step = max(1, len(paths) // 20)
        for file_idx, path in enumerate(paths, start=1):
            if file_idx == 1 or file_idx % progress_step == 0 or file_idx == len(paths):
                print(f"Teacher fuzzy: fronte {file_idx}/{len(paths)}")
            try:
                with open(path, encoding="utf-8") as handle:
                    document = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"File ignorato {path}: {exc}")
                continue
            scenario = document.get("scenario") or {}
            ac0 = scenario.get("ac0") or {}
            direct_distance = dist_diretta_2d(ac0.get("p0"), ac0.get("p_target"))
            if direct_distance is None or direct_distance <= 0.0:
                continue
            solutions: list[dict[str, Any]] = []
            preferences: list[float] = []
            features: list[np.ndarray] = []
            for source_idx, raw in enumerate(document.get("population") or []):
                try:
                    path_length, h, bearing = self.fuzzy_decider._extract_solution_metrics(raw)
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                real = raw.get("metriche_reali") or {}
                collision = bool(
                    raw.get("collisione") or raw.get("conflict")
                    or real.get("collisione") or real.get("conflict")
                )
                solution = {
                    "horizontal_target_error_nm": h,
                    "path_length": path_length,
                    "final_bearing_error_deg": bearing,
                    "collisione": collision,
                    "genotype": raw.get("genome", []),
                    "source_idx": source_idx,
                }
                preference = self.fuzzy_decider.evaluate_solution(
                    path_length, h, bearing, dist_diretta=direct_distance
                )
                solutions.append(solution)
                preferences.append(preference)
                features.append(self._scale_route_metrics(solution, direct_distance))
            if not solutions:
                continue
            pref_array = np.asarray(preferences, dtype=np.float32)
            feature_array = np.asarray(features, dtype=np.float32)
            if not (len(solutions) == len(pref_array) == len(feature_array)):
                raise RuntimeError(f"Disallineamento durante il caricamento di {path}")
            best_idx = int(np.argmax(pref_array))
            filename = os.path.basename(path)
            result.append({
                "scenario_id": len(result),
                "filename": filename,
                "group_id": self.scenario_group(filename),
                "pareto_front": solutions,
                "fuzzy_preferences": pref_array,
                "policy_features": feature_array,
                "best_idx": best_idx,
                "max_preference": float(pref_array[best_idx]),
            })
        if skipped:
            print(f"Soluzioni ignorate per metriche mancanti/non valide: {skipped}")
        return result

    @staticmethod
    def _scale_route_metrics(solution: dict[str, Any], direct_distance: float | None) -> np.ndarray:
        if solution["collisione"]:
            return np.ones(len(ROUTE_FEATURES), dtype=np.float32)
        path = solution["path_length"]
        ratio = path / direct_distance if direct_distance and direct_distance > 0.0 else 1.0
        return np.asarray([
            (np.clip(ratio, 1.0, 2.5) - 1.0) / 1.5,
            np.clip(solution["horizontal_target_error_nm"], 0.0, 0.10) / 0.10,
            np.clip(abs(solution["final_bearing_error_deg"]), 0.0, 180.0) / 180.0,
        ], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "scenario_idx" in options:
            scenario_idx = int(options["scenario_idx"]) % len(self.scenarios)
        elif self.scenario_sampling == "cycle":
            scenario_idx = self._cycle_idx % len(self.scenarios)
            self._cycle_idx += 1
        else:
            scenario_idx = int(self.np_random.integers(len(self.scenarios)))
        canonical = self.scenarios[scenario_idx]
        n_routes = len(canonical["pareto_front"])
        # This permutation is mandatory augmentation, including evaluation.
        permutation = (
            self.np_random.permutation(n_routes)
            if self.permute_candidates else np.arange(n_routes, dtype=np.int64)
        )
        preferences = canonical["fuzzy_preferences"][permutation]
        features = canonical["policy_features"][permutation]
        routes = [canonical["pareto_front"][int(i)] for i in permutation]
        best_idx = int(np.argmax(preferences))
        self._action_permutation = permutation
        self.current_scenario = {
            **canonical,
            "pareto_front": routes,
            "fuzzy_preferences": preferences,
            "policy_features": features,
            "best_idx": best_idx,
            "max_preference": float(preferences[best_idx]),
            "action_permutation": permutation.copy(),
        }
        self._assert_alignment()
        return self._get_observation(), {
            "scenario_id": canonical.get("scenario_id", scenario_idx),
            "filename": canonical["filename"],
            "action_permutation": permutation.copy(),
        }

    def _assert_alignment(self) -> None:
        if self.current_scenario is None:
            raise RuntimeError("Scenario corrente assente")
        current = self.current_scenario
        lengths = (len(current["pareto_front"]), len(current["fuzzy_preferences"]),
                   len(current["policy_features"]), len(current["action_permutation"]))
        if len(set(lengths)) != 1:
            raise RuntimeError(f"Array disallineati in {current['filename']}: {lengths}")
        if current["best_idx"] != int(np.argmax(current["fuzzy_preferences"])):
            raise RuntimeError(f"best_idx disallineato in {current['filename']}")

    def _get_observation(self) -> np.ndarray:
        if self.current_scenario is None:
            raise RuntimeError("Chiamare reset() prima di _get_observation()")
        padded = np.full((self.max_pareto_size, len(ROUTE_FEATURES)), PADDING_VALUE, dtype=np.float32)
        features = self.current_scenario["policy_features"]
        padded[:len(features)] = features
        return padded.reshape(-1)

    def action_masks(self) -> np.ndarray:
        if self.current_scenario is None:
            raise RuntimeError("Chiamare reset() prima di action_masks()")
        mask = np.zeros(self.max_pareto_size, dtype=bool)
        mask[:len(self.current_scenario["pareto_front"])] = True
        return mask

    def step(self, action: Any):
        if self.current_scenario is None:
            raise RuntimeError("Chiamare reset() prima di step()")
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError(f"Azione discreta attesa, ricevuta shape={action_array.shape}")
        action_idx = int(action_array.reshape(-1)[0])
        n_valid = len(self.current_scenario["pareto_front"])
        if action_idx < 0 or action_idx >= n_valid:
            raise ValueError(f"Azione invalida {action_idx}; rotte valide={n_valid}")
        preferences = self.current_scenario["fuzzy_preferences"]
        maximum = float(self.current_scenario["max_preference"])
        minimum = float(np.min(preferences))
        selected = float(preferences[action_idx])
        regret = max(0.0, maximum - selected)
        span = maximum - minimum
        # Only fuzzy-equivalent optima are rewarded; every wrong index is negative.
        if regret <= MATCH_TOLERANCE:
            reward = 1.0
        else:
            reward = -(0.25 + 0.75 * regret / span)
        best_idx = int(self.current_scenario["best_idx"])
        permutation = self.current_scenario["action_permutation"]
        info = {
            "suggested_idx": action_idx,
            "chosen_idx": best_idx,
            "suggested_source_idx": int(permutation[action_idx]),
            "chosen_source_idx": int(permutation[best_idx]),
            "match": regret <= MATCH_TOLERANCE,
            "suggested_preference": selected,
            "chosen_preference": float(preferences[best_idx]),
            "max_preference": maximum,
            "fuzzy_deviation": regret,
            "all_preferences": preferences.tolist(),
        }
        return self._get_observation(), float(reward), True, False, info


class MetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.matches: list[float] = []
        self.regrets: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "match" in info:
                self.matches.append(float(info["match"]))
                self.regrets.append(float(info["fuzzy_deviation"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.verbose and self.matches:
            print(
                f"Bandit train, ultime 500: accuracy={np.mean(self.matches[-500:]):.3f}, "
                f"regret={np.mean(self.regrets[-500:]):.3f}"
            )


class BestFuzzyValidationCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        max_pareto_size: int,
        training_seed: int | None,
        validation_fraction: float,
        validation_env: MultiScenarioEnv,
        validation_groups: Sequence[str],
        eval_freq: int,
    ) -> None:
        super().__init__()
        self.save_path = save_path
        self.max_pareto_size = max_pareto_size
        self.training_seed = training_seed
        self.validation_fraction = validation_fraction
        self.validation_env = validation_env
        self.validation_groups = list(validation_groups)
        self.eval_freq = max(1, eval_freq)
        self.best_accuracy = -math.inf
        self.best_regret = math.inf
        self.best_timestep = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        self.evaluate_and_save()
        return True

    def evaluate_and_save(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        validation_results = _evaluate_environment(
            self.model,
            self.validation_env,
            split_name="validation",
            seed=None,
            n_shuffles=1,
            verbose=False,
        )
        validation_summary = _summarize_results(
            validation_results,
            n_files=len(self.validation_env.scenarios),
            n_shuffles=1,
        )
        accuracy = validation_summary["accuracy_strict"]
        regret = validation_summary["mean_regret"]
        improved = accuracy > self.best_accuracy or (
            np.isclose(accuracy, self.best_accuracy) and regret < self.best_regret
        )
        print(
            f"[validation @ {self.num_timesteps}] accuracy={accuracy:.2f}% | "
            f"regret={regret:.4f}" + (" | NUOVO BEST" if improved else "")
        )
        if not improved:
            return
        self.best_accuracy = accuracy
        self.best_regret = regret
        self.best_timestep = int(self.model.num_timesteps)
        self.model.save(os.path.join(self.save_path, "best_model"))
        stats = {
            "stats_version": 2,
            "timesteps": self.best_timestep,
            "validation_accuracy_strict": validation_summary["accuracy_strict"],
            "validation_accuracy_operational": validation_summary["accuracy_operational"],
            "validation_mean_fuzzy_distance": validation_summary["mean_regret"],
            "validation_median_fuzzy_distance": validation_summary["median_regret"],
            "validation_max_fuzzy_distance": validation_summary["max_regret"],
            "validation_num_files": validation_summary["num_files"],
            "strict_tolerance": validation_summary["strict_tolerance"],
            "operational_tolerance": validation_summary["operational_tolerance"],
        }
        with open(
            os.path.join(self.save_path, "best_model_stats.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(stats, handle, indent=2)
        _save_contract(
            self.save_path,
            self.max_pareto_size,
            self.model.num_timesteps,
            None,
            training_seed=self.training_seed,
            validation_fraction=self.validation_fraction,
            validation_groups=self.validation_groups,
        )


def _save_contract(
    folder: str,
    max_pareto_size: int,
    timesteps: int,
    mean_reward: float | None,
    *,
    training_seed: int | None,
    validation_fraction: float,
    validation_groups: Sequence[str],
) -> None:
    os.makedirs(folder, exist_ok=True)
    payload = {
        "contract_version": 5,
        "problem": "contextual_bandit",
        "gamma": 0.0,
        "max_pareto_size": int(max_pareto_size),
        "max_supported_pareto_size": MAX_PARETO_SIZE,
        "route_features": list(ROUTE_FEATURES),
        "padding": -1.0,
        "action_shuffling": True,
        "training_seed": None if training_seed is None else int(training_seed),
        "validation_fraction": float(validation_fraction),
        "validation_groups": sorted(validation_groups),
        "timesteps": int(timesteps),
        "validation_mean_reward": mean_reward,
    }
    with open(os.path.join(folder, "model_contract.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _grouped_train_validation_split(
    scenarios: Sequence[dict[str, Any]], validation_fraction: float, seed: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        groups[scenario["group_id"]].append(scenario)
    group_ids = np.asarray(sorted(groups), dtype=object)
    if validation_fraction <= 0.0 or len(group_ids) < 2:
        return list(scenarios), []
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    n_validation = min(len(group_ids) - 1, max(1, round(len(group_ids) * validation_fraction)))
    validation_ids = set(group_ids[:n_validation])
    train = [s for s in scenarios if s["group_id"] not in validation_ids]
    validation = [s for s in scenarios if s["group_id"] in validation_ids]
    return train, validation


split_train_validation = _grouped_train_validation_split


def _split_from_validation_groups(
    scenarios: Sequence[dict[str, Any]], validation_groups: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_ids = set(validation_groups)
    available_ids = {item["group_id"] for item in scenarios}
    missing = sorted(validation_ids - available_ids)
    if missing:
        raise ValueError(
            f"Il contratto del modello riferisce {len(missing)} gruppi validation assenti "
            f"nel dataset corrente: {', '.join(missing[:10])}"
        )
    return (
        [item for item in scenarios if item["group_id"] not in validation_ids],
        [item for item in scenarios if item["group_id"] in validation_ids],
    )


def _assert_external_test_is_holdout(input_folder: str) -> None:
    """Fail fast if a previous pruning run left a physical scenario in both sets."""
    normalized = os.path.normpath(input_folder)
    test_folder = (
        os.path.join(os.path.dirname(normalized), "test")
        if os.path.basename(normalized) == "train"
        else os.path.join(normalized, "test")
    )
    if not os.path.isdir(test_folder):
        return
    train_files = glob.glob(os.path.join(input_folder, "*_pareto_front_*_pruned.json"))
    test_files = glob.glob(os.path.join(test_folder, "*_pareto_front_*_pruned.json"))
    train_groups = {MultiScenarioEnv.scenario_group(os.path.basename(path)) for path in train_files}
    test_groups = {MultiScenarioEnv.scenario_group(os.path.basename(path)) for path in test_files}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        preview = ", ".join(overlap[:10])
        raise ValueError(
            f"Data leakage: {len(overlap)} scenari fisici compaiono sia in train sia in test "
            f"({preview}). Rigenerare lo split GA/pruning prima del training."
        )


def train_from_folder(
    input_folder: str = "./data/pruned/train",
    total_timesteps: int = 200_000,
    *,
    validation_fraction: float = 0.20,
    seed: int | None = None,
    output_dir: str = ".",
) -> MaskablePPO:
    """Train without touching the external test folder."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps deve essere positivo")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction deve essere in [0, 1)")
    _assert_external_test_is_holdout(input_folder)
    oracle = FuzzyDecisionMaker()
    # Il contratto del nuovo checkpoint e' fisso a 100 azioni; le rotte non
    # presenti sono sempre mascherate. I vecchi checkpoint restano valutabili
    # passando esplicitamente la loro capacita' letta dall'action_space.
    full_dataset = MultiScenarioEnv(
        input_folder, oracle, max_pareto_size=MAX_PARETO_SIZE
    )
    train_scenarios, validation_scenarios = _grouped_train_validation_split(
        full_dataset.scenarios, validation_fraction, seed
    )
    train_groups = sorted({item["group_id"] for item in train_scenarios})
    validation_groups = sorted({item["group_id"] for item in validation_scenarios})
    raw_train = full_dataset.subset(train_scenarios, sampling="random")
    train_env = DummyVecEnv([lambda: raw_train])
    callbacks: list[BaseCallback] = [MetricsCallback(verbose=1)]

    best_dir = os.path.join(output_dir, "best_model")
    best_callback: BestFuzzyValidationCallback | None = None
    if validation_scenarios:
        raw_validation = MultiScenarioEnv(
            None,
            oracle,
            max_pareto_size=full_dataset.max_pareto_size,
            scenarios=validation_scenarios,
            scenario_sampling="cycle",
            permute_candidates=False,
        )
        best_callback = BestFuzzyValidationCallback(
            best_dir,
            full_dataset.max_pareto_size,
            training_seed=seed,
            validation_fraction=validation_fraction,
            validation_env=raw_validation,
            validation_groups=validation_groups,
            eval_freq=5000,
        )
        callbacks.append(best_callback)
        print(
            f"Split per scenario fisico: train={len(train_groups)} gruppi/"
            f"{len(train_scenarios)} fronti, validation={len(validation_groups)} gruppi/"
            f"{len(validation_scenarios)} fronti; il test non viene caricato."
        )

    model = MaskablePPO(
        ParetoSetPolicy,
        train_env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.0,
        gae_lambda=1.0,
        ent_coef=0.001,
        vf_coef=0.10,
        max_grad_norm=0.5,
        target_kl=0.03,
        seed=seed,
        device="cpu",
        verbose=1,
        policy_kwargs={"ortho_init": True},
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    if best_callback is not None:
        best_callback.evaluate_and_save()
    final_path = os.path.join(output_dir, "pareto_recommender_bandit_final")
    model.save(final_path)
    _save_contract(
        output_dir,
        full_dataset.max_pareto_size,
        model.num_timesteps,
        None,
        training_seed=seed,
        validation_fraction=validation_fraction,
        validation_groups=validation_groups,
    )
    if not validation_scenarios:
        os.makedirs(best_dir, exist_ok=True)
        model.save(os.path.join(best_dir, "best_model"))
        _save_contract(
            best_dir,
            full_dataset.max_pareto_size,
            model.num_timesteps,
            None,
            training_seed=seed,
            validation_fraction=validation_fraction,
            validation_groups=validation_groups,
        )
    print(f"Training concluso. Modello finale: {final_path}.zip")
    print(f"Miglior modello fuzzy-validation: {os.path.join(best_dir, 'best_model')}.zip")
    return model


def _validate_model_environment(model: MaskablePPO, env: MultiScenarioEnv) -> None:
    if not isinstance(model.action_space, spaces.Discrete):
        raise TypeError("Il modello caricato non usa uno spazio azioni Discrete")
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Contratto osservazione incompatibile: modello={model.observation_space.shape}, "
            f"dataset={env.observation_space.shape}. Riaddestrare con il refactoring corrente."
        )


def _evaluate_environment(
    model: MaskablePPO,
    env: MultiScenarioEnv,
    *,
    split_name: str,
    seed: int | None,
    n_shuffles: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    _validate_model_environment(model, env)
    results: list[dict[str, Any]] = []
    for shuffle_idx in range(n_shuffles):
        for scenario_idx in range(len(env.scenarios)):
            reset_seed = (
                None if seed is None
                else seed + shuffle_idx * len(env.scenarios) + scenario_idx
            )
            observation, _ = env.reset(seed=reset_seed, options={"scenario_idx": scenario_idx})
            mask = env.action_masks()
            action, _ = model.predict(
                observation[np.newaxis, :],
                action_masks=mask[np.newaxis, :],
                deterministic=True,
            )
            _, reward, _, _, info = env.step(action)
            current = env.current_scenario
            assert current is not None
            preferences = current["fuzzy_preferences"]
            maximum = float(current["max_preference"])
            permutation = current["action_permutation"]
            solution_distances = []
            for slot, (source_idx, preference) in enumerate(zip(permutation, preferences)):
                distance = max(0.0, maximum - float(preference))
                solution_distances.append({
                    "source_idx": int(source_idx),
                    "shuffled_slot": int(slot),
                    "fuzzy_preference": float(preference),
                    "distance_from_file_max": distance,
                    "selected_by_model": int(source_idx) == info["suggested_source_idx"],
                    "is_fuzzy_argmax": int(slot) == info["chosen_idx"],
                    "is_fuzzy_equivalent": distance <= MATCH_TOLERANCE,
                })
            solution_distances.sort(key=lambda item: item["source_idx"])
            item = {
                "split": split_name,
                "group_id": current["group_id"],
                "filename": current["filename"],
                "shuffle": shuffle_idx,
                "n_routes": int(mask.sum()),
                "reward": float(reward),
                "file_max_fuzzy_preference": maximum,
                "solution_distances_from_max": solution_distances,
                **info,
            }
            results.append(item)
            if verbose and n_shuffles == 1:
                status = "MATCH" if info["match"] else f"regret={info['fuzzy_deviation']:.2f}"
                print(
                    f"[{split_name:10s}] {scenario_idx + 1:03d}/{len(env.scenarios):03d} "
                    f"{item['filename']} n={item['n_routes']:2d} {status}"
                )
    return results


def _percentage(values: np.ndarray) -> float:
    return float(100.0 * values.mean()) if values.size else float("nan")


def _summarize_results(
    results: Sequence[dict[str, Any]],
    *,
    n_files: int,
    n_shuffles: int,
    strict_tolerance: float = MATCH_TOLERANCE,
    operational_tolerance: float = MATCH_TOLERANCE,
) -> dict[str, Any]:
    if not results:
        raise ValueError("Nessun risultato da riassumere")
    regrets = np.asarray([item["fuzzy_deviation"] for item in results], dtype=float)
    route_counts = np.asarray([item["n_routes"] for item in results], dtype=int)
    strict = regrets <= strict_tolerance
    operational = regrets <= operational_tolerance
    nontrivial = route_counts > 1

    per_group_raw: dict[str, list[int]] = defaultdict(list)
    per_file_raw: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(results):
        per_group_raw[item["group_id"]].append(index)
        per_file_raw[item["filename"]].append(index)

    per_group = []
    for group_id, indices in sorted(per_group_raw.items()):
        selected = np.asarray(indices, dtype=int)
        filenames = sorted({results[index]["filename"] for index in indices})
        per_group.append({
            "group_id": group_id,
            "num_files": len(filenames),
            "accuracy_strict": _percentage(strict[selected]),
            "accuracy_operational": _percentage(operational[selected]),
            "mean_fuzzy_regret": float(regrets[selected].mean()),
            "max_fuzzy_regret": float(regrets[selected].max()),
            "fully_correct_strict": bool(strict[selected].all()),
            "fully_correct_operational": bool(operational[selected].all()),
            "majority_correct_strict": bool(strict[selected].mean() >= 0.5),
        })

    regret_stable = []
    source_action_stable = []
    for indices in per_file_raw.values():
        file_regrets = np.asarray([results[index]["fuzzy_deviation"] for index in indices])
        source_actions = {results[index]["suggested_source_idx"] for index in indices}
        regret_stable.append(float(file_regrets.max() - file_regrets.min()) <= 1e-6)
        source_action_stable.append(len(source_actions) == 1)

    worst_index = int(np.argmax(regrets))
    return {
        # Compatibility aliases used by the previous test-only evaluator.
        "num_scenarios": int(n_files),
        "accuracy": _percentage(strict),
        "mean_fuzzy_regret": float(regrets.mean()),
        "max_fuzzy_regret": float(regrets.max()),
        # Full final-comparison metrics.
        "num_files": int(n_files),
        "num_physical_groups": len(per_group),
        "n_shuffles": int(n_shuffles),
        "num_evaluations": len(results),
        "strict_tolerance": float(strict_tolerance),
        "operational_tolerance": float(operational_tolerance),
        "accuracy_strict": _percentage(strict),
        "accuracy_operational": _percentage(operational),
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "p95_regret": float(np.percentile(regrets, 95)),
        "max_regret": float(regrets.max()),
        "worst_case": {
            "filename": results[worst_index]["filename"],
            "group_id": results[worst_index]["group_id"],
            "regret": float(regrets[worst_index]),
            "n_routes": int(results[worst_index]["n_routes"]),
        },
        "nontrivial_fronts": {
            "num_evaluations": int(nontrivial.sum()),
            "accuracy_strict": _percentage(strict[nontrivial]),
            "accuracy_operational": _percentage(operational[nontrivial]),
            "mean_regret": float(regrets[nontrivial].mean()) if nontrivial.any() else None,
        },
        "physical_groups": {
            "fully_correct_strict": int(sum(item["fully_correct_strict"] for item in per_group)),
            "fully_correct_operational": int(sum(item["fully_correct_operational"] for item in per_group)),
            "majority_correct_strict": int(sum(item["majority_correct_strict"] for item in per_group)),
            "details": per_group,
        },
        "permutation_consistency": {
            "regret_stable_files_pct": 100.0 * float(np.mean(regret_stable)),
            "source_action_stable_files_pct": 100.0 * float(np.mean(source_action_stable)),
        },
        "detailed_results": list(results),
    }


def _read_best_model_stats(model_path: str) -> dict[str, Any]:
    parent = os.path.dirname(os.path.abspath(model_path)) or os.getcwd()
    stats_path = os.path.join(parent, "best_model_stats.json")
    if not os.path.isfile(stats_path):
        return {}
    try:
        with open(stats_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _print_best_model_header(
    model_path: str,
    validation_summary: dict[str, Any] | None = None,
) -> None:
    stats = _read_best_model_stats(model_path)
    accuracy = (
        validation_summary["accuracy_strict"]
        if validation_summary is not None
        else stats.get("validation_accuracy_strict")
    )
    operational_accuracy = (
        validation_summary["accuracy_operational"]
        if validation_summary is not None
        else stats.get("validation_accuracy_operational")
    )
    mean_distance = (
        validation_summary["mean_regret"]
        if validation_summary is not None
        else stats.get("validation_mean_fuzzy_distance")
    )
    print("\n" + "=" * 78)
    print("BEST MODEL SELEZIONATO PER IL TEST")
    print(f"Checkpoint: {model_path}")
    if stats.get("stats_version") == 2 and stats.get("timesteps") is not None:
        print(f"Timestep del best checkpoint: {stats['timesteps']}")
    if accuracy is not None and mean_distance is not None:
        print(f"Accuracy validation strict: {float(accuracy):.2f}%")
        if operational_accuracy is not None:
            print(f"Accuracy validation operativa: {float(operational_accuracy):.2f}%")
        print(f"Distanza fuzzy media dal migliore (validation): {float(mean_distance):.4f}")
    else:
        print(
            "Metriche validation del best checkpoint non presenti: saranno disponibili "
            "dopo un nuovo training o tramite il confronto finale."
        )
    print("=" * 78 + "\n")


def _print_solution_distances(results: Sequence[dict[str, Any]]) -> None:
    """Print every route distance once, using shuffle zero and original JSON indices."""
    first_shuffle = [item for item in results if item["shuffle"] == 0]
    print("\nDISTANZA DI OGNI SOLUZIONE DAL MASSIMO FUZZY DEL PROPRIO FILE")
    print("(indici 'source' = posizione originale nel JSON; PPO/FUZZY marcano le scelte)\n")
    for item in first_shuffle:
        print(
            f"{item['filename']} | n={item['n_routes']} | "
            f"max fuzzy={item['file_max_fuzzy_preference']:.4f}"
        )
        print(f"  {'source':>6} {'slot':>5} {'preferenza':>11} {'distanza':>10}  stato")
        for solution in item["solution_distances_from_max"]:
            markers = []
            if solution["selected_by_model"]:
                markers.append("PPO")
            if solution["is_fuzzy_argmax"]:
                markers.append("FUZZY-MAX")
            elif solution["is_fuzzy_equivalent"]:
                markers.append("FUZZY-EQUIV")
            print(
                f"  {solution['source_idx']:>6d} {solution['shuffled_slot']:>5d} "
                f"{solution['fuzzy_preference']:>11.4f} "
                f"{solution['distance_from_file_max']:>10.4f}  "
                f"{','.join(markers) if markers else '-'}"
            )
        print()


def _print_comparison_table(summaries: dict[str, dict[str, Any]]) -> None:
    reference = next(iter(summaries.values()))
    strict_label = f"acc@{reference['strict_tolerance']:.2f}"
    operational_label = f"acc@{reference['operational_tolerance']:.2f}"
    print("\nCONFRONTO FINALE DELLO STESSO CHECKPOINT")
    print(
        f"{'split':<11} {'file':>5} {'gruppi':>7} {strict_label:>10} "
        f"{operational_label:>10} {'regret':>9} {'p95':>8} {'max':>8} {'acc n>1':>9}"
    )
    print("-" * 93)
    for split_name in ("train", "validation", "test"):
        summary = summaries.get(split_name)
        if summary is None:
            continue
        print(
            f"{split_name:<11} {summary['num_files']:>5d} {summary['num_physical_groups']:>7d} "
            f"{summary['accuracy_strict']:>9.2f}% {summary['accuracy_operational']:>9.2f}% "
            f"{summary['mean_regret']:>9.3f} {summary['p95_regret']:>8.3f} "
            f"{summary['max_regret']:>8.3f} "
            f"{summary['nontrivial_fronts']['accuracy_strict']:>8.2f}%"
        )


def compare_train_validation_test(
    model_path: str,
    *,
    data_folder: str = "./data/pruned/train",
    test_folder: str = "./data/pruned/test",
    validation_fraction: float = 0.20,
    training_seed: int | None = None,
    evaluation_seed: int | None = None,
    n_shuffles: int = 5,
    operational_tolerance: float = MATCH_TOLERANCE,
    verbose: bool = False,
    show_solution_distances: bool = True,
) -> dict[str, Any]:
    """Compare one checkpoint on the exact train/validation split and hold-out test."""
    if n_shuffles <= 0:
        raise ValueError("n_shuffles deve essere positivo")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction deve essere in [0, 1)")
    if operational_tolerance < MATCH_TOLERANCE:
        raise ValueError("operational_tolerance non puo' essere inferiore alla soglia stretta")
    _assert_external_test_is_holdout(data_folder)
    model_parent = os.path.dirname(os.path.abspath(model_path)) or os.getcwd()
    contract_path = os.path.join(model_parent, "model_contract.json")
    contract: dict[str, Any] = {}
    if os.path.isfile(contract_path):
        with open(contract_path, encoding="utf-8") as handle:
            contract = json.load(handle)
        saved_seed = contract.get("training_seed")
        saved_fraction = contract.get("validation_fraction")
        if training_seed is not None and saved_seed is not None and int(saved_seed) != training_seed:
            raise ValueError(
                f"training_seed={training_seed} non coincide con il contratto del modello "
                f"({saved_seed})"
            )
        if saved_fraction is not None and not np.isclose(float(saved_fraction), validation_fraction):
            raise ValueError(
                f"validation_fraction={validation_fraction} non coincide con il contratto "
                f"del modello ({saved_fraction})"
            )
    model = MaskablePPO.load(model_path, device="cpu")
    if not isinstance(model.action_space, spaces.Discrete):
        raise TypeError("Il modello caricato non usa uno spazio azioni Discrete")
    capacity = int(model.action_space.n)
    oracle = FuzzyDecisionMaker()
    full_dataset = MultiScenarioEnv(
        data_folder, oracle, max_pareto_size=capacity, scenario_sampling="cycle"
    )
    if "validation_groups" in contract:
        train_scenarios, validation_scenarios = _split_from_validation_groups(
            full_dataset.scenarios, contract["validation_groups"]
        )
    else:
        train_scenarios, validation_scenarios = _grouped_train_validation_split(
            full_dataset.scenarios, validation_fraction, training_seed
        )
    split_environments = {
        "train": full_dataset.subset(train_scenarios, sampling="cycle"),
        "test": MultiScenarioEnv(
            test_folder,
            oracle,
            max_pareto_size=capacity,
            scenario_sampling="cycle",
        ),
    }
    if validation_scenarios:
        split_environments["validation"] = full_dataset.subset(
            validation_scenarios, sampling="cycle"
        )

    summaries: dict[str, dict[str, Any]] = {}
    seed_offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
    for split_name in ("train", "validation", "test"):
        env = split_environments.get(split_name)
        if env is None:
            continue
        if split_name == "test":
            _print_best_model_header(model_path, summaries.get("validation"))
        results = _evaluate_environment(
            model,
            env,
            split_name=split_name,
            seed=(
                None if evaluation_seed is None
                else evaluation_seed + seed_offsets[split_name]
            ),
            n_shuffles=n_shuffles,
            verbose=verbose,
        )
        summaries[split_name] = _summarize_results(
            results,
            n_files=len(env.scenarios),
            n_shuffles=n_shuffles,
            operational_tolerance=operational_tolerance,
        )

    train_summary = summaries["train"]
    test_summary = summaries["test"]
    validation_summary = summaries.get("validation")
    comparison = {
        "model_path": model_path,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "validation_fraction": validation_fraction,
        "n_shuffles": n_shuffles,
        "model_contract": contract or None,
        "splits": summaries,
        "generalization_gaps_percentage_points": {
            "train_minus_test_strict": float(
                train_summary["accuracy_strict"] - test_summary["accuracy_strict"]
            ),
            "train_minus_test_operational": float(
                train_summary["accuracy_operational"] - test_summary["accuracy_operational"]
            ),
            "validation_minus_test_strict": (
                float(validation_summary["accuracy_strict"] - test_summary["accuracy_strict"])
                if validation_summary else None
            ),
        },
    }
    if show_solution_distances:
        _print_solution_distances(test_summary["detailed_results"])
    _print_comparison_table(summaries)
    gaps = comparison["generalization_gaps_percentage_points"]
    print(
        f"Gap train-test: {gaps['train_minus_test_strict']:+.2f} pp (strict), "
        f"{gaps['train_minus_test_operational']:+.2f} pp (operational)."
    )
    print(
        f"Worst test: {test_summary['worst_case']['filename']} | "
        f"regret={test_summary['worst_case']['regret']:.3f}."
    )
    test_groups = test_summary["physical_groups"]
    print(
        f"Gruppi test completamente corretti: "
        f"{test_groups['fully_correct_strict']}/{test_summary['num_physical_groups']} strict, "
        f"{test_groups['fully_correct_operational']}/{test_summary['num_physical_groups']} operational."
    )
    consistency = test_summary["permutation_consistency"]
    print(
        f"Consistenza su {n_shuffles} permutazioni: "
        f"regret stabile={consistency['regret_stable_files_pct']:.2f}%, "
        f"rotta originale stabile={consistency['source_action_stable_files_pct']:.2f}%."
    )
    print(
        f"RISULTATO FINALE TEST: accuracy strict={test_summary['accuracy_strict']:.2f}% | "
        f"accuracy operativa={test_summary['accuracy_operational']:.2f}% | "
        f"distanza fuzzy media={test_summary['mean_regret']:.4f}"
    )
    return comparison


def evaluate_model(
    folder_test: str,
    model_path: str,
    *,
    seed: int | None = None,
    n_shuffles: int = 1,
    verbose: bool = True,
    show_solution_distances: bool = True,
) -> dict[str, Any]:
    """Backward-compatible exhaustive evaluation of the hold-out folder only."""
    if n_shuffles <= 0:
        raise ValueError("n_shuffles deve essere positivo")
    model = MaskablePPO.load(model_path, device="cpu")
    if not isinstance(model.action_space, spaces.Discrete):
        raise TypeError("Il modello caricato non usa uno spazio azioni Discrete")
    if verbose:
        _print_best_model_header(model_path)
    env = MultiScenarioEnv(
        folder_test,
        FuzzyDecisionMaker(),
        max_pareto_size=int(model.action_space.n),
        scenario_sampling="cycle",
    )
    results = _evaluate_environment(
        model,
        env,
        split_name="test",
        seed=seed,
        n_shuffles=n_shuffles,
        verbose=verbose,
    )
    summary = _summarize_results(
        results, n_files=len(env.scenarios), n_shuffles=n_shuffles
    )
    if verbose:
        if show_solution_distances:
            _print_solution_distances(results)
        print(
            f"RISULTATO FINALE TEST: accuracy strict={summary['accuracy_strict']:.2f}% | "
            f"accuracy operativa={summary['accuracy_operational']:.2f}% | "
            f"distanza fuzzy media={summary['mean_regret']:.4f}"
        )
    return summary


def test_generalizzazione(folder_test: str, model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible public name for the exhaustive evaluator."""
    return evaluate_model(folder_test, model_path, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", default="data/pruned/train")
    train.add_argument("--timesteps", type=int, default=600_000)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-dir", default=".")
    test = commands.add_parser("test")
    test.add_argument("--data", default="data/pruned/test")
    test.add_argument("--model", default="best_model/best_model.zip")
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--shuffles", type=int, default=5)
    compare = commands.add_parser("compare")
    compare.add_argument("--model", default="best_model/best_model.zip")
    compare.add_argument("--train", default="data/pruned/train")
    compare.add_argument("--test", default="data/pruned/test")
    compare.add_argument("--training-seed", type=int, default=42)
    compare.add_argument("--evaluation-seed", type=int, default=42)
    compare.add_argument("--shuffles", type=int, default=5)
    args = parser.parse_args()
    if args.command == "train":
        train_from_folder(
            args.data, args.timesteps,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.command == "test":
        evaluate_model(
            args.data, args.model, seed=args.seed, n_shuffles=args.shuffles
        )
    else:
        compare_train_validation_test(
            args.model,
            data_folder=args.train,
            test_folder=args.test,
            training_seed=args.training_seed,
            evaluation_seed=args.evaluation_seed,
            n_shuffles=args.shuffles,
        )


if __name__ == "__main__":
    main()
