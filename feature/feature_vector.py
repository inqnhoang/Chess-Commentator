# features/feature_vector.py
from enums.board import Color, PieceType, GamePhase
from enums.eval import (
    KingSafety,
    CenterControl,
    PawnStructureHealth,
    PieceActivity,
    TacticalDanger,
    EvaluationBucket,
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
        self.pawn_structure: PawnStructureHealth | None = None
        self.piece_activity: PieceActivity | None = None
        self.tactical_danger: TacticalDanger | None = None
        self.evaluation_bucket: EvaluationBucket | None = None

        # Move/Threat analysis
        self.move_impact: MoveImpact | None = None
        self.discovered_attack_or_check: DiscoveredTactic | None = None
        self.hanging_piece: bool | None = None
        self.promotion_threat: bool | None = None
        self.open_files_toward_king: int | None = None
        self.rooks_on_open_files: int | None = None

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in vars(self).items())
