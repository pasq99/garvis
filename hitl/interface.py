
from __future__ import annotations

from typing import Any, Sequence


def show_options(
    *, sequence: int, total: int, filename: str, active_fitness: str,
    options: Sequence[dict[str, Any]],
) -> None:
    print(f"\n[{sequence}/{total}] {filename} | fitness attiva: {active_fitness}")
    print("Scelta  PPO  rank   f1 path(NM)   f2 errore(NM)   f3 bearing(deg)")
    for option in options:
        fitness = option["fitness"]
        marker = "SI" if option["is_ppo"] else ""
        print(
            f"  {option['display_id']:<3}   {marker:<3}  {option['active_rank']:>4}"
            f"   {fitness['f1']:>11.5f}   {fitness['f2']:>13.6f}   {fitness['f3']:>15.5f}"
        )


def ask_controller(options: Sequence[dict[str, Any]]) -> int:
    valid = {option["display_id"]: option["action_idx"] for option in options}
    while True:
        answer = input(f"Scelta corretta del controllore ({'/'.join(valid)}, q=esci): ").strip().upper()
        if answer == "Q":
            raise KeyboardInterrupt
        if answer in valid:
            return valid[answer]
        print("Scelta non valida.")


def show_controller_label(option: dict[str, Any], category: str) -> None:
    confirmation = "conferma PPO" if option["is_ppo"] else "override PPO"
    print(
        f"> LABEL CORRETTO CONTROLLORE: {option['display_id']} "
        f"(source_idx={option['source_idx']}, {category}, {confirmation})"
    )
