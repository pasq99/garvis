
from __future__ import annotations

import argparse
import os
import random
import uuid
from typing import Any, Sequence

from hitl.choices import (
    DECISION_PROFILES,
    FITNESSES,
    alternative_indices,
    classify_decision,
    decision_index,
    mixed_schedule,
    parse_mix,
    ranked_indices,
)
from hitl.dataset import DatasetWriter, utc_now
from hitl.interface import ask_controller, show_controller_label, show_options
from hitl.scenarios import load_scenarios, model_input


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pruned/test")
    parser.add_argument("--model", default="best_model/best_model.zip")
    parser.add_argument("--output", default="hitl/datasets/initial_hitl.jsonl")
    parser.add_argument("--samples", type=int, default=300, help="0 usa tutti i fronti eleggibili")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--fitness-cycle", default="f1,f2,f3")
    parser.add_argument(
        "--decision", choices=("human", "mixed", *DECISION_PROFILES), default="human",
        help="Scelta umana, profilo fisso o mix sintetico",
    )
    parser.add_argument("--mix", default="ppo=25,best=25,median=25,worst=25")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Non mostra le tabelle in modalita' automatica")
    return parser


def _candidate_payload(
    action_idx: int,
    scenario: dict[str, Any],
    rank_by_action: dict[int, int],
    display_id: str,
    ppo_idx: int,
) -> dict[str, Any]:
    route = scenario["pareto_front"][action_idx]
    values = (
        float(route["path_length"]),
        float(route["horizontal_target_error_nm"]),
        float(route["final_bearing_error_deg"]),
    )
    return {
        "display_id": display_id,
        "action_idx": action_idx,
        "source_idx": int(route["source_idx"]),
        "is_ppo": action_idx == ppo_idx,
        "active_rank": rank_by_action[action_idx] + 1,
        "fitness": {"f1": values[0], "f2": values[1], "f3": values[2]},
        "policy_features": [float(value) for value in scenario["policy_features"][action_idx]],
    }


def _decision_schedule(
    count: int,
    block_size: int,
    decision: str,
    mix: dict[str, float],
    rng: random.Random,
) -> list[str]:
    if decision != "mixed":
        return [decision] * count
    result: list[str] = []
    for start in range(0, count, block_size):
        result.extend(mixed_schedule(min(block_size, count - start), mix, rng))
    return result


def collect(args: argparse.Namespace) -> int:
    import numpy as np
    from sb3_contrib import MaskablePPO

    if args.samples < 0 or args.block_size <= 0:
        raise ValueError("--samples deve essere >= 0 e --block-size deve essere positivo")
    fitness_cycle = [item.strip() for item in args.fitness_cycle.split(",") if item.strip()]
    unknown = [item for item in fitness_cycle if item not in FITNESSES]
    if not fitness_cycle or unknown:
        raise ValueError(f"--fitness-cycle non valido: {args.fitness_cycle}")
    mix = parse_mix(args.mix)

    model = MaskablePPO.load(args.model, device="cpu")
    capacity = int(model.action_space.n)
    scenarios = load_scenarios(args.data)
    eligible = [index for index, scenario in enumerate(scenarios) if len(scenario["pareto_front"]) >= 5]
    scenario_rng = random.Random(args.seed)
    scenario_rng.shuffle(eligible)
    count = len(eligible) if args.samples == 0 else args.samples
    if count > len(eligible):
        raise ValueError(f"Richiesti {count} fronti, ma solo {len(eligible)} hanno almeno 5 rotte")
    selected_scenarios = eligible[:count]
    decision_modes = _decision_schedule(
        count, args.block_size, args.decision, mix, random.Random(args.seed + 1)
    )
    option_rng = random.Random(args.seed + 2)
    session_id = str(uuid.uuid4())
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "created_at": utc_now(),
        "data": os.path.abspath(args.data),
        "model": os.path.abspath(args.model),
        "samples": count,
        "block_size": args.block_size,
        "fitness_cycle": fitness_cycle,
        "fitness_direction": "minimize",
        "decision": args.decision,
        "mix": mix if args.decision == "mixed" else None,
        "seed": args.seed,
        "candidate_set_size": 5,
        "correct_label_field": "correct_controller_choice.option_index",
        "excluded_fronts_with_fewer_than_5_routes": len(scenarios) - len(eligible),
    }

    interrupted = False
    writer: DatasetWriter | None = None
    try:
        with DatasetWriter(args.output, manifest) as writer:
            for offset, (scenario_idx, requested_profile) in enumerate(
                zip(selected_scenarios, decision_modes), start=1
            ):
                active_key = fitness_cycle[((offset - 1) // args.block_size) % len(fitness_cycle)]
                fitness_index = int(FITNESSES[active_key]["index"])
                scenario = scenarios[scenario_idx]
                observation, mask = model_input(scenario, capacity)
                action, _ = model.predict(
                    observation[np.newaxis, :],
                    action_masks=mask[np.newaxis, :],
                    deterministic=True,
                )
                ppo_idx = int(np.asarray(action).reshape(-1)[0])
                fitnesses = [
                    (
                        route["path_length"],
                        route["horizontal_target_error_nm"],
                        route["final_bearing_error_deg"],
                    )
                    for route in scenario["pareto_front"]
                ]
                order = ranked_indices(fitnesses, fitness_index)
                ranks = {candidate: rank for rank, candidate in enumerate(order)}
                shown = alternative_indices(fitnesses, ppo_idx, fitness_index)
                option_rng.shuffle(shown)
                options = [
                    _candidate_payload(candidate, scenario, ranks, chr(65 + display_idx), ppo_idx)
                    for display_idx, candidate in enumerate(shown)
                ]
                if not args.quiet or requested_profile == "human":
                    show_options(
                        sequence=offset,
                        total=count,
                        filename=scenario["filename"],
                        active_fitness=active_key,
                        options=options,
                    )
                controller_idx = (
                    ask_controller(options)
                    if requested_profile == "human"
                    else decision_index(requested_profile, fitnesses, ppo_idx, fitness_index)
                )
                chosen_option = next(option for option in options if option["action_idx"] == controller_idx)
                chosen_option_index = options.index(chosen_option)
                category = classify_decision(controller_idx, fitnesses, ppo_idx, fitness_index)
                if not args.quiet or requested_profile == "human":
                    show_controller_label(chosen_option, category)
                writer.append({
                    "sequence": offset,
                    "block_index": (offset - 1) // args.block_size,
                    "active_fitness": {
                        "key": active_key,
                        **FITNESSES[active_key],
                        "direction": "minimize",
                    },
                    "scenario": {
                        "filename": scenario["filename"],
                        "group_id": scenario["group_id"],
                        "num_routes": len(fitnesses),
                    },
                    "options": options,
                    "correct_controller_choice": {
                        "option_index": chosen_option_index,
                        "action_idx": controller_idx,
                        "source_idx": int(scenario["pareto_front"][controller_idx]["source_idx"]),
                        "display_id": chosen_option["display_id"],
                        "requested_profile": requested_profile,
                        "fitness_position": category,
                        "confirms_ppo": controller_idx == ppo_idx,
                        "is_training_target": True,
                    },
                })
    except KeyboardInterrupt:
        interrupted = True

    saved = 0 if writer is None else writer.records
    status = "interrotta" if interrupted else "completata"
    print(f"Sessione {status}: {saved}/{count} decisioni salvate in {args.output}")
    return 130 if interrupted else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return collect(args)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
