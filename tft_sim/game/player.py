from dataclasses import dataclass, field
from typing import List, Optional
from tft_sim.game.units import Unit

@dataclass
class Player:
    health:       int   = 100
    gold:         int   = 0
    level:        int   = 1
    xp:           int   = 0
    win_streak:   int   = 0
    loss_streak:  int   = 0
    board:        List[Optional[Unit]]  = field(default_factory=lambda: [None] * 10)
    bench:        List[Optional[Unit]]  = field(default_factory=lambda: [None] * 9)
    current_shop: List[Optional[int]]   = field(default_factory=lambda: [None] * 5)
    is_agent:     bool  = False
    is_eliminated: bool = False
