from enum import Enum, auto

class KingSafety(Enum):
    SAFE = auto()
    EXPOSED = auto()
    UNDER_ATTACK = auto()

class CenterControl(Enum):
    WHITE = auto()
    BLACK = auto()
    CONTESTED = auto()

class PawnStructureHealth(Enum):
    POOR = auto()
    DECENT = auto()
    GOOD = auto()

class PieceActivity(Enum):
    LOW = auto()
    EQUAL = auto()
    HIGH = auto()

class TacticalDanger(Enum):
    NONE = auto()
    THREAT = auto()
    FORCING = auto()

class EvaluationBucket(Enum):
    WINNING = auto()
    BETTER = auto()
    EQUAL = auto()
