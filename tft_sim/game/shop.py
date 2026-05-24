from collections import defaultdict
from typing import List, Optional

import numpy as np

POOL_SIZES = {
    1: 45,
    2: 30,
    3: 25,
    4: 18,
    5: 10
}

SHOP_ODDS = {
    1: {1: 1.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
    2: {1: 1.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
    3: {1: 0.75, 2: 0.25, 3: 0.00, 4: 0.00, 5: 0.00},
    4: {1: 0.55, 2: 0.30, 3: 0.15, 4: 0.00, 5: 0.00},
    5: {1: 0.45, 2: 0.33, 3: 0.20, 4: 0.02, 5: 0.00},
    6: {1: 0.30, 2: 0.40, 3: 0.25, 4: 0.05, 5: 0.00},
    7: {1: 0.19, 2: 0.30, 3: 0.35, 4: 0.15, 5: 0.01},
    8: {1: 0.15, 2: 0.20, 3: 0.35, 4: 0.24, 5: 0.06},
    9: {1: 0.10, 2: 0.15, 3: 0.30, 4: 0.30, 5: 0.15},
}

class PoolManager:
    def __init__(self, unit_db):
        self.pool = defaultdict(list)
        # Initialize pool
        for unit_id, data in unit_db.unit_data.items():
            cost = data['cost']
            amount = POOL_SIZES.get(cost, 0)
            self.pool[cost].extend([unit_id] * amount)
            
    def get_available(self, cost_tier: int) -> List[int]:
        return self.pool.get(cost_tier, [])
        
    def reserve(self, unit_id: int, cost_tier: int) -> bool:
        """Removes a unit from the pool when it appears in a shop."""
        if unit_id in self.pool[cost_tier]:
            self.pool[cost_tier].remove(unit_id)
            return True
        return False
        
    def return_unit(self, unit_id: int, cost_tier: int):
        """Returns a unit back to the pool."""
        self.pool[cost_tier].append(unit_id)

    def return_units(self, unit_ids: List[int], unit_db):
        """Helper to return multiple units (e.g. when rerolling or selling)."""
        for uid in unit_ids:
            if uid is not None:
                cost = unit_db.get_unit_base_data(uid)['cost']
                self.return_unit(uid, cost)

def sample_cost_tier(level: int, rng: Optional[np.random.Generator] = None) -> int:
    odds = SHOP_ODDS.get(level, SHOP_ODDS[9])
    if rng is None:
        rng = np.random.default_rng()
    r = float(rng.random())
    cumulative = 0.0
    for cost, prob in odds.items():
        cumulative += prob
        if r <= cumulative:
            return cost
    return 1  # Fallback

def roll_shop(player, pool_manager, unit_db, rng: Optional[np.random.Generator] = None) -> List[int]:
    if rng is None:
        rng = np.random.default_rng()
    shop = []
    for _ in range(5):
        cost_tier = sample_cost_tier(player.level, rng)
        available = pool_manager.get_available(cost_tier)
        if available:
            unit_id = int(rng.choice(available))
            pool_manager.reserve(unit_id, cost_tier)
            shop.append(unit_id)
        else:
            shop.append(None)
    return shop

def reroll(player, pool_manager, unit_db, rng: Optional[np.random.Generator] = None):
    if player.gold < 2:
        return
    pool_manager.return_units(player.current_shop, unit_db)
    player.gold -= 2
    player.current_shop = roll_shop(player, pool_manager, unit_db, rng)
