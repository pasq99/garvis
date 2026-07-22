
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Sequence

from recommender.policy_contract import ROUTE_FEATURES


def load_feedback(
    paths: Sequence[str],
) -> tuple[
    list[list[list[float]]], list[int], list[str], list[tuple[str, str]], dict[str, Any]
]:
    features: list[list[list[float]]] = []
    targets: list[int] = []
    groups: list[str] = []
    strata: list[tuple[str, str]] = []
    profiles: Counter[str] = Counter()
    fitnesses: Counter[str] = Counter()
    confirms_ppo = 0

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"Feedback JSONL non trovato: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    options = record["options"]
                    choice = record["correct_controller_choice"]
                    target = int(choice["option_index"])
                    group_id = str(record["scenario"]["group_id"])
                    filename = str(record["scenario"]["filename"])
                    active_fitness = str(record["active_fitness"]["key"])
                    requested_profile = str(choice.get("requested_profile", "human"))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Record non valido in {path}:{line_number}: {exc}") from exc
                if choice.get("is_training_target", True) is not True:
                    continue
                filename_group = re.match(r"((?:exp_)?\d+)_pareto_front_", filename)
                if filename_group is None or filename_group.group(1) != group_id:
                    raise ValueError(f"group_id e filename disallineati in {path}:{line_number}")
                if len(options) != 5 or not 0 <= target < 5:
                    raise ValueError(f"Servono 5 opzioni e un target valido in {path}:{line_number}")
                if int(options[target]["action_idx"]) != int(choice["action_idx"]):
                    raise ValueError(f"option_index e action_idx disallineati in {path}:{line_number}")

                row: list[list[float]] = []
                for option in options:
                    values = [float(value) for value in option["policy_features"]]
                    if len(values) != len(ROUTE_FEATURES) or not all(math.isfinite(value) for value in values):
                        raise ValueError(f"policy_features non valide in {path}:{line_number}")
                    row.append(values)
                features.append(row)
                targets.append(target)
                groups.append(group_id)
                strata.append((active_fitness, requested_profile))
                profiles[requested_profile] += 1
                fitnesses[active_fitness] += 1
                confirms_ppo += int(bool(choice.get("confirms_ppo")))

    if not features:
        raise ValueError("Nessun target HITL utilizzabile")
    return features, targets, groups, strata, {
        "records": len(features),
        "physical_groups": len(set(groups)),
        "profiles": dict(sorted(profiles.items())),
        "fitnesses": dict(sorted(fitnesses.items())),
        "confirms_ppo": confirms_ppo,
    }


def split_by_group(
    groups: Sequence[str],
    validation_fraction: float,
    seed: int,
    strata: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Split physical scenarios while balancing fitness/profile record strata."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction deve essere in [0, 1)")
    if strata is not None and len(strata) != len(groups):
        raise ValueError("groups e strata devono avere la stessa lunghezza")
    unique_groups = sorted(set(groups))
    if validation_fraction == 0.0:
        return list(range(len(groups))), [], unique_groups, []
    if len(unique_groups) < 2:
        raise ValueError("Servono almeno due group_id per creare la validation HITL")

    labels = list(strata) if strata is not None else [("all", "all")] * len(groups)
    indices_by_group: dict[str, list[int]] = {group: [] for group in unique_groups}
    for index, group in enumerate(groups):
        indices_by_group[group].append(index)
    target_records = max(1, round(len(groups) * validation_fraction))
    target_strata = {
        label: count * validation_fraction for label, count in Counter(labels).items()
    }
    rng = random.Random(seed)
    best: tuple[tuple[Any, ...], list[str]] | None = None
    candidates = unique_groups.copy()
    for _ in range(20_000):
        rng.shuffle(candidates)
        selected: list[str] = []
        records = 0
        for group in candidates:
            candidate_records = records + len(indices_by_group[group])
            if records < target_records and (
                candidate_records <= target_records
                or target_records - records >= candidate_records - target_records
            ):
                selected.append(group)
                records = candidate_records
            elif records >= target_records:
                break
        counts = Counter(
            labels[index] for group in selected for index in indices_by_group[group]
        )
        errors = [abs(counts[label] - target) for label, target in target_strata.items()]
        score = (
            abs(records - target_records),
            max(errors),
            sum(error * error for error in errors),
            tuple(sorted(selected)),
        )
        if best is None or score < best[0]:
            best = score, selected.copy()
    assert best is not None
    validation_groups = sorted(best[1])
    validation_set = set(validation_groups)
    train_indices = [index for index, group in enumerate(groups) if group not in validation_set]
    validation_indices = [index for index, group in enumerate(groups) if group in validation_set]
    train_groups = sorted(set(groups) - validation_set)
    return train_indices, validation_indices, train_groups, validation_groups


def load_fixed_split(
    groups: Sequence[str], path: str
) -> tuple[list[int], list[int], list[str], list[str]]:
    split_path = Path(path)
    try:
        with split_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        train_groups = sorted(str(group) for group in payload["train_groups"])
        validation_groups = sorted(str(group) for group in payload["validation_groups"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Split HITL non valido: {split_path}: {exc}") from exc
    available = set(groups)
    declared = set(train_groups) | set(validation_groups)
    if set(train_groups) & set(validation_groups):
        raise ValueError("Lo split HITL contiene gruppi sia in train sia in validation")
    if not train_groups:
        raise ValueError("Lo split HITL non contiene gruppi train")
    if declared != available:
        missing = sorted(available - declared)
        extra = sorted(declared - available)
        raise ValueError(
            f"Split HITL incompatibile: gruppi mancanti={missing[:5]}, estranei={extra[:5]}"
        )
    validation_set = set(validation_groups)
    return (
        [index for index, group in enumerate(groups) if group not in validation_set],
        [index for index, group in enumerate(groups) if group in validation_set],
        train_groups,
        validation_groups,
    )


def actor_metrics(
    route_encoder: Any,
    route_scorer: Any,
    features: Any,
    targets: Any,
    indices: Any,
) -> dict[str, float] | None:
    import torch as th
    from torch.nn import functional as F

    if len(indices) == 0:
        return None
    with th.no_grad():
        logits = route_scorer(route_encoder(features[indices])).squeeze(-1)
        selected = logits.gather(1, targets[indices].unsqueeze(1)).squeeze(1)
        alternatives = logits.clone()
        alternatives.scatter_(1, targets[indices].unsqueeze(1), -th.inf)
        return {
            "loss": float(F.cross_entropy(logits, targets[indices])),
            "accuracy": 100.0 * float((logits.argmax(1) == targets[indices]).float().mean()),
            "mean_target_margin": float((selected - alternatives.max(1).values).mean()),
        }


def optimize_actor(
    route_encoder: Any,
    route_scorer: Any,
    features: Any,
    targets: Any,
    train_indices: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
    seed: int,
) -> list[float]:
    import torch as th
    from torch.nn import functional as F

    parameters = list(route_encoder.parameters()) + list(route_scorer.parameters())
    optimizer = th.optim.Adam(parameters, lr=learning_rate)
    generator = th.Generator().manual_seed(seed)
    losses: list[float] = []
    route_encoder.train()
    route_scorer.train()
    for _ in range(epochs):
        permutation = train_indices[th.randperm(len(train_indices), generator=generator)]
        total_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            logits = route_scorer(route_encoder(features[indices])).squeeze(-1)
            loss = F.cross_entropy(logits, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        losses.append(total_loss / len(permutation))
    route_encoder.eval()
    route_scorer.eval()
    return losses


def next_model_directory(root: Path) -> tuple[int, Path]:
    root.mkdir(parents=True, exist_ok=True)
    versions = [
        int(match.group(1))
        for path in root.iterdir()
        if path.is_dir() and (match := re.fullmatch(r"ppo_(\d+)", path.name))
    ]
    version = max(versions, default=0) + 1
    return version, root / f"ppo_{version}"


def _feedback_digest(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for raw_path in paths:
        with Path(raw_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def fine_tune(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch as th
    from sb3_contrib import MaskablePPO
    import recommender.rlfuzzy  # noqa: F401 - registers the serialized custom policy

    if (
        args.epochs <= 0 or args.batch_size <= 0
        or args.learning_rate <= 0 or args.max_grad_norm <= 0
    ):
        raise ValueError("Epochs, batch size, learning rate e max grad norm devono essere positivi")
    raw_features, raw_targets, groups, strata, dataset_stats = load_feedback(args.feedback)
    if args.split_file:
        train_indices, validation_indices, train_groups, validation_groups = load_fixed_split(
            groups, args.split_file
        )
    else:
        train_indices, validation_indices, train_groups, validation_groups = split_by_group(
            groups, args.validation_fraction, args.split_seed, strata
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    model = MaskablePPO.load(args.model, device="cpu")
    extractor = model.policy.mlp_extractor
    route_encoder = getattr(extractor, "route_encoder", None)
    route_scorer = getattr(extractor, "route_scorer", None)
    if route_encoder is None or route_scorer is None:
        raise TypeError("Il modello non usa il ParetoSetExtractor richiesto dal fine-tuning HITL")

    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    actor_parameters = list(route_encoder.parameters()) + list(route_scorer.parameters())
    for parameter in actor_parameters:
        parameter.requires_grad_(True)

    x = th.tensor(raw_features, dtype=th.float32)
    y = th.tensor(raw_targets, dtype=th.long)
    train_tensor = th.tensor(train_indices, dtype=th.long)
    validation_tensor = th.tensor(validation_indices, dtype=th.long)
    route_encoder.eval()
    route_scorer.eval()
    initial_train = actor_metrics(route_encoder, route_scorer, x, y, train_tensor)
    initial_validation = actor_metrics(route_encoder, route_scorer, x, y, validation_tensor)
    assert initial_train is not None
    epoch_losses = optimize_actor(
        route_encoder, route_scorer, x, y, train_tensor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    final_train = actor_metrics(route_encoder, route_scorer, x, y, train_tensor)
    final_validation = actor_metrics(route_encoder, route_scorer, x, y, validation_tensor)
    assert final_train is not None

    version, output_dir = next_model_directory(Path(args.output_dir))
    output_dir.mkdir()
    model_path = output_dir / "model"
    model.save(model_path)

    base_contract_path = Path(args.model).resolve().parent / "model_contract.json"
    contract: dict[str, Any] = {}
    if base_contract_path.is_file():
        with base_contract_path.open(encoding="utf-8") as handle:
            contract = json.load(handle)
    fine_tuning_metadata = {
        "model_index": version,
        "model_label": f"PPO^{version}",
        "base_model": str(Path(args.model).resolve()),
        "feedback": [str(Path(path).resolve()) for path in args.feedback],
        "feedback_sha256": _feedback_digest(args.feedback),
        "replay": False,
        "objective": "controller_choice_cross_entropy_over_5_options",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "validation_fraction": args.validation_fraction,
        "split_seed": args.split_seed,
        "split_file": None if args.split_file is None else str(Path(args.split_file).resolve()),
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "dataset": dataset_stats,
    }
    contract["fine_tuning"] = fine_tuning_metadata
    with (output_dir / "model_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)

    with (output_dir / "hitl_split.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "train_groups": train_groups,
            "validation_groups": validation_groups,
        }, handle, indent=2, ensure_ascii=False)

    report = {
        **fine_tuning_metadata,
        "model_path": str((model_path.with_suffix(".zip")).resolve()),
        "split": {
            "train_records": len(train_indices),
            "validation_records": len(validation_indices),
            "train_physical_groups": len(train_groups),
            "validation_physical_groups": len(validation_groups),
            "train_strata": {
                "/".join(label): count
                for label, count in sorted(Counter(strata[index] for index in train_indices).items())
            },
            "validation_strata": {
                "/".join(label): count
                for label, count in sorted(
                    Counter(strata[index] for index in validation_indices).items()
                )
            },
        },
        "metrics": {
            "train": {"before": initial_train, "after": final_train},
            "validation": {"before": initial_validation, "after": final_validation},
        },
        # Compatibility aliases for reports generated by the first implementation.
        "initial_loss": initial_train["loss"],
        "final_loss": final_train["loss"],
        "initial_accuracy": initial_train["accuracy"],
        "final_accuracy": final_train["accuracy"],
        "epoch_losses": epoch_losses,
    }
    with (output_dir / "fine_tuning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Salvato {report['model_label']} in {report['model_path']}")
    print(
        f"HITL train: accuracy {initial_train['accuracy']:.2f}% -> "
        f"{final_train['accuracy']:.2f}% | loss {initial_train['loss']:.4f} -> "
        f"{final_train['loss']:.4f}"
    )
    if initial_validation is not None and final_validation is not None:
        print(
            f"HITL validation: accuracy {initial_validation['accuracy']:.2f}% -> "
            f"{final_validation['accuracy']:.2f}% | loss {initial_validation['loss']:.4f} -> "
            f"{final_validation['loss']:.4f}"
        )
    return model_path.with_suffix(".zip")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback", nargs="+", default=["hitl/datasets/initial_hitl.jsonl"],
        help="Uno o piu' JSONL HITL da unire nel fine-tuning",
    )
    parser.add_argument("--model", default="best_model/best_model/best_model.zip")
    parser.add_argument("--output-dir", default="fine_tuning/models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=700)
    parser.add_argument(
        "--split-file",
        help="hitl_split.json di un run precedente, per riusare gli stessi gruppi",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        fine_tune(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
