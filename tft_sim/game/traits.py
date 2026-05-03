from collections import defaultdict

def compute_active_traits(board_units, trait_definitions):
    """
    Computes active traits given a list of units and the global trait definitions.
    """
    trait_counts = defaultdict(int)
    for unit in board_units:
        if unit is None: continue
        for trait in unit.traits:
            trait_counts[trait] += 1

    active_bonuses = {}
    for trait, count in trait_counts.items():
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
