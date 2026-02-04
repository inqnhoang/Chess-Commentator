from state.game_state import GameState
from feature_extractor import FeatureExtractor
from delta_vector import DeltaVector
from chess import Move

class DataPoint:
    def __init__(self, curr: GameState, move: Move, next: GameState):
        self.curr = curr
        self.move = move
        self.next = next
        self.delta_vector: DeltaVector | None = None

    def compute_deltas(self, engine):
        curr_features = FeatureExtractor.extract(self.curr, engine=engine)
        next_features = FeatureExtractor.extract(self.next, engine=engine)

        # convert None to 0 for numeric fields
        def safe(val):
            return 0 if val is None else val
        
        def changed(curr_val, next_val):
            if curr_val != next_val:
                return next_val
            return None

        stockfish_eval_delta = safe(next_features.sf_evaluation) - safe(curr_features.sf_evaluation)
        mate_in_delta = safe(next_features.mate_in) - safe(curr_features.mate_in)
        material_balance_delta = safe(next_features.material_balance) - safe(curr_features.material_balance)
        piece_activity_delta = safe(next_features.piece_activity) - safe(curr_features.piece_activity)
        pawn_structure_delta = safe(next_features.pawn_structure) - safe(curr_features.pawn_structure)
        tactical_danger_delta = safe(next_features.tactical_danger) - safe(curr_features.tactical_danger)
        rooks_on_open_files_delta = safe(next_features.rooks_on_open_files) - safe(curr_features.rooks_on_open_files)
        open_files_toward_king_delta = safe(next_features.open_files_toward_king) - safe(curr_features.open_files_toward_king)
        win_percentage_delta = safe(next_features.win_percentage) - safe(curr_features.win_percentage)

        game_phase_delta = changed(curr_features.game_phase, next_features.game_phase)
        king_safety_delta = changed(curr_features.king_safety, next_features.king_safety)
        center_control_delta = changed(curr_features.center_control, next_features.center_control)
        evaluation_bucket_delta = changed(curr_features.evaluation_bucket, next_features.evaluation_bucket)
        move_impact_delta = changed(curr_features.move_impact, next_features.move_impact)
        discovered_attack_or_check_delta = changed(curr_features.discovered_attack_or_check, next_features.discovered_attack_or_check)
        hanging_piece_delta = changed(curr_features.hanging_piece, next_features.hanging_piece)
        promotion_threat_delta = changed(curr_features.promotion_threat, next_features.promotion_threat)

        self.delta_vector = DeltaVector(  # keep as empty instance if using dataclass
            material_balance_delta,
            piece_activity_delta,
            pawn_structure_delta,
            tactical_danger_delta,
            stockfish_eval_delta,
            mate_in_delta,
            rooks_on_open_files_delta,
            open_files_toward_king_delta,
            win_percentage_delta,

            game_phase_delta,
            king_safety_delta,
            center_control_delta,
            evaluation_bucket_delta,
            move_impact_delta,
            discovered_attack_or_check_delta,
            hanging_piece_delta,
            promotion_threat_delta
        )

        return self.delta_vector