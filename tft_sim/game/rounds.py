"""Round type resolution (see docs/tft_rl_spec.md)."""
from typing import Literal

RoundType = Literal["carousel", "pve_creep", "pvp"]


def determine_round_type(stage: int, round_in_stage: int) -> RoundType:
    if stage == 1 and round_in_stage == 1:
        return "carousel"
    if stage == 1 and 2 <= round_in_stage <= 4:
        return "pve_creep"
    if stage >= 2 and round_in_stage == 1:
        return "pve_creep"
    return "pvp"
