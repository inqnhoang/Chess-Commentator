# features/feature_vector.py
from enums.board import Color, PieceType, GamePhase
from enums.eval import (
    KingSafety,
    CenterControl,
    EvaluationBucket,
)
from enums.moves import (
    MoveImpact,
    DiscoveredTactic
)

class FeatureVector:
    """Holds all 15 features derived from a GameState."""

    def __init__(self):
        # Game state
        self.side_to_move: Color | None = None
        self.game_phase: GamePhase | None = None

        # Board evaluation
        self.material_balance: int | None = None
        self.king_safety: KingSafety | None = None
        self.center_control: CenterControl | None = None
        self.pawn_structure: int| None = None
        self.piece_activity: int | None = None
        self.tactical_danger: int | None = None
        self.evaluation_bucket: EvaluationBucket | None = None

        # Move/Threat analysis
        self.move_impact: MoveImpact | None = None
        self.discovered_attack_or_check: DiscoveredTactic | None = None
        self.hanging_piece: bool | None = None
        self.promotion_threat: bool | None = None
        self.open_files_toward_king: int | None = None
        self.rooks_on_open_files: int | None = None
        
        # Stock fish analysis
        self.mate_in: int | None = None
        self.sf_evaluation: int | None = None

        # Win percentages w/ Lichess Formula
        self.win_percentage: float | None = None

        # Bools
        self.is_capture: bool = None
        self.is_check: bool = None
        self.is_checkmate: bool = None
        self.is_castle: bool = None
        self.is_en_passant: bool = None
        self.is_promotion: bool = None
        self.piece_moved: str | None = None
        self.captured_piece: str | None = None
        self.promotion_piece: str | None = None
        
    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in vars(self).items())
