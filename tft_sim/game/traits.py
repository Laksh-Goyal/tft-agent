from collections import defaultdict
from typing import Dict, List, Optional

def trait_counts(board_units) -> Dict[str, int]:
    """Count trait occurrences on board (non-None units only)."""
    counts: Dict[str, int] = defaultdict(int)
    for unit in board_units:
        if unit is None:
            continue
        for trait in unit.traits:
            counts[trait] += 1
    return dict(counts)


def active_breakpoint_level(trait: str, count: int, trait_definitions: dict) -> int:
    """Highest active breakpoint for trait at given count, or 0 if none."""
    if trait not in trait_definitions or count <= 0:
        return 0
    breakpoints = trait_definitions[trait]["breakpoints"]
    active = [bp for bp in breakpoints if bp <= count]
    return max(active) if active else 0


def detect_new_breakpoints(
    before: Dict[str, int],
    after: Dict[str, int],
    trait_definitions: dict,
) -> List[str]:
    """Traits whose active breakpoint level increased."""
    new_traits: List[str] = []
    all_traits = set(before.keys()) | set(after.keys())
    for trait in all_traits:
        old_bp = active_breakpoint_level(trait, before.get(trait, 0), trait_definitions)
        new_bp = active_breakpoint_level(trait, after.get(trait, 0), trait_definitions)
        if new_bp > old_bp:
            new_traits.append(trait)
    return new_traits


def compute_active_traits(board_units, trait_definitions):
    """
    Computes active traits given a list of units and the global trait definitions.
    """
    counts = trait_counts(board_units)
    active_bonuses = {}
    for trait, count in counts.items():
        if trait not in trait_definitions:
            continue
        breakpoints = trait_definitions[trait]["breakpoints"]
        effects     = trait_definitions[trait]["effects"]
        
        # find highest active breakpoint
        active_bp = max((bp for bp in breakpoints if bp <= count), default=None)
        if active_bp:
            # Trait effects keys are strings in JSON, e.g. "2", "4", "6"
            active_bonuses[trait] = effects[str(active_bp)]

    return active_bonuses

def next_breakpoint(trait, count, trait_definitions):
    """
    Returns the next breakpoint for a given trait.
    """
    if trait not in trait_definitions:
        return None
    breakpoints = trait_definitions[trait]["breakpoints"]
    upcoming = [bp for bp in breakpoints if bp > count]
    return min(upcoming) if upcoming else None


def max_breakpoint(trait: str, trait_definitions: dict) -> int:
    if trait not in trait_definitions:
        return 1
    bps = trait_definitions[trait]["breakpoints"]
    return max(bps) if bps else 1


def synergy_obs_pair(trait: str, count: int, trait_definitions: dict) -> tuple[float, float]:
    """Spec format: [count_normalized, breakpoint_progress]."""
    max_bp = max_breakpoint(trait, trait_definitions)
    count_norm = min(1.0, count / max_bp) if max_bp > 0 else 0.0
    nxt = next_breakpoint(trait, count, trait_definitions)
    if nxt is None:
        progress = 1.0
    elif count <= 0:
        progress = 0.0
    else:
        prev_bp = active_breakpoint_level(trait, count, trait_definitions)
        span = nxt - (prev_bp if prev_bp > 0 else 0)
        progress = min(1.0, (count - (prev_bp if prev_bp > 0 else 0)) / span) if span > 0 else 1.0
    return count_norm, progress
