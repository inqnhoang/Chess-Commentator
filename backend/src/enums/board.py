from enum import Enum, auto

class Color(Enum):
    WHITE = auto()
    BLACK = auto()

class PieceType(Enum):
    PAWN = auto()
    KNIGHT = auto()
    BISHOP = auto()
    ROOK = auto()
    QUEEN = auto()
    KING = auto()

class GamePhase(Enum):
    OPENING = auto()
    MIDDLEGAME = auto()
    ENDGAME = auto()
    