from enums.board import GamePhase
from enums.eval import KingSafety, CenterControl, EvaluationBucket
from enums.moves import MoveImpact, DiscoveredTactic

class DeltaVector:
    def __init__(
        self,
        material_balance_delta: float,
        piece_activity_delta: float,
        pawn_structure_delta: float,
        tactical_danger_delta: float,
        stockfish_eval_delta: float,
        mate_in_delta: int,
        rooks_on_open_files_delta: int,
        open_files_toward_king_delta: int,
        win_percentage_delta: float,

        game_phase_delta: GamePhase | None,
        king_safety_delta: KingSafety | None,
        center_control_delta: CenterControl | None,
        evaluation_bucket_delta: EvaluationBucket | None,
        move_impact_delta: MoveImpact | None,
        discovered_attack_or_check_delta: DiscoveredTactic | None,
        hanging_piece_delta: bool | None,
        promotion_threat_delta: bool | None
    ):
        self.material_balance_delta = material_balance_delta
        self.piece_activity_delta = piece_activity_delta
        self.pawn_structure_delta = pawn_structure_delta
        self.tactical_danger_delta = tactical_danger_delta
        self.stockfish_eval_delta = stockfish_eval_delta
        self.mate_in_delta = mate_in_delta
        self.rooks_on_open_files_delta = rooks_on_open_files_delta
        self.open_files_toward_king_delta = open_files_toward_king_delta
        self.win_percentage_delta = win_percentage_delta

        self.game_phase_delta = game_phase_delta
        self.king_safety_delta = king_safety_delta
        self.center_control_delta = center_control_delta
        self.evaluation_bucket_delta = evaluation_bucket_delta
        self.move_impact_delta = move_impact_delta
        self.discovered_attack_or_check_delta = discovered_attack_or_check_delta
        self.hanging_piece_delta = hanging_piece_delta
        self.promotion_threat_delta = promotion_threat_delta
