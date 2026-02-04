from enum import Enum, auto

class KingSafety(Enum):
    SAFE = auto()
    EXPOSED = auto()
    UNDER_ATTACK = auto()

class CenterControl(Enum):
    WHITE = auto()
    BLACK = auto()
    CONTESTED = auto()

class EvaluationBucket(Enum):
    WINNING = auto()
    BETTER = auto()
    EQUAL = auto()
