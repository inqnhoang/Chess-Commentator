from enum import Enum, auto

class MoveImpact(Enum):
    IMPROVES = auto()
    NEUTRAL = auto()
    WORSENS = auto()

class DiscoveredTactic(Enum):
    NONE = auto()
    ATTACK = auto()
    CHECK = auto()
    