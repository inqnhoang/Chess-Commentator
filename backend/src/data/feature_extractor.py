from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import chess
import chess.engine
import math



PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
]



def logistic_win_pct(cp: int) -> float:
    cp = max(-2000, min(2000, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def opponent(color: chess.Color) -> chess.Color:
    return not color


def count_bits(x) -> int:
    try:
        return len(x)
    except TypeError:
        return chess.popcount(int(x))


def squares_of(bb: chess.Bitboard) -> List[chess.Square]:
    return list(chess.SquareSet(bb))


# engine analysis container
@dataclass
class EngineLine:
    cp: Optional[int]   # centipawns
    mate: Optional[int]   # mate in N
    pv: List[str]   # UCI moves


@dataclass
class MoveEngineInfo:
    before_best: Optional[EngineLine]   # eval of the before position
    after_best: Optional[EngineLine]   # eval after engine's best move from before position
    after_played: Optional[EngineLine]   # eval after the actually played move


# feature output
@dataclass
class MoveFeatures:
    # context
    before_fen: str
    after_fen: str
    move_uci: str
    mover: str   # "white"/"black"
    ply: Optional[int] = None

    # core engine
    eval_before_cp: Optional[int] = None
    eval_after_cp: Optional[int] = None
    eval_swing_cp: Optional[int] = None   # after - before
    played_vs_best_loss_cp: Optional[int] = None   # (after_best - after_played) (positive = no mistake)
    mate_before: Optional[int] = None
    mate_after: Optional[int] = None
    winpct_before: Optional[float] = None
    winpct_after: Optional[float] = None
    pv_best: Optional[List[str]] = None   # PV from before (best line)
    pv_after: Optional[List[str]] = None   # pv from after played move

    # forcing / tactical events
    is_capture: bool = False
    captured_piece: Optional[str] = None
    is_check: bool = False
    is_double_check: bool = False
    is_mate: bool = False
    gives_discovered_attack: bool = False
    is_promotion: bool = False
    promoted_to: Optional[str] = None
    is_castle: bool = False
    is_en_passant: bool = False

    # trades / simplification
    queen_trade: bool = False
    major_trade: bool = False   # capture involved rook/queen
    simplification_score: int = 0   # decrease in total non-pawn material (both sides)

    # per-side deltas of remaining pieces (after - before)
    delta_queens_mover: int = 0
    delta_rooks_mover: int = 0
    delta_bishops_mover: int = 0
    delta_knights_mover: int = 0

    delta_queens_opp: int = 0
    delta_rooks_opp: int = 0
    delta_bishops_opp: int = 0
    delta_knights_opp: int = 0

    # trade/exchange flags
    rook_trade: bool = False
    bishop_trade: bool = False
    knight_trade: bool = False
    minor_trade: bool = False   # bishop or knight trade
    exchange_trade: bool = False   # rook for minor piece, exchange sacrifice
    bishop_for_knight_imbalance: bool = False
    knight_for_bishop_imbalance: bool = False

    # capture category
    captured_major: bool = False   # queen, rook
    captured_minor: bool = False   # knight, bishop

    # trade offer flags
    queen_trade_available: bool = False
    rook_trade_available: bool = False
    bishop_trade_available: bool = False
    knight_trade_available: bool = False

    queen_trade_offered: bool = False
    rook_trade_offered: bool = False
    bishop_trade_offered: bool = False
    knight_trade_offered: bool = False

    minor_trade_offered: bool = False
    major_trade_offered: bool = False

    # material changes
    material_delta_mover: int = 0   # in pawn units, positive = mover gained
    net_exchange_value: int = 0   # placeholder

    # hanging pieces / tactics
    hanging_value_created_for_opponent: int = 0
    hanging_value_created_for_self: int = 0
    hanging_highest_piece_self: Optional[str] = None
    hanging_highest_square_self: Optional[str] = None
    hanging_highest_piece_opponent: Optional[str] = None
    hanging_highest_square_opponent: Optional[str] = None

    # king safety
    king_in_check_before: bool = False
    king_in_check_after: bool = False
    king_zone_attack_delta: int = 0
    open_files_near_enemy_king_delta: int = 0
    castling_rights_lost: bool = False
    exposed_king_delta: int = 0

    # activity / development / space
    mobility_delta_mover: int = 0
    mobility_delta_opponent: int = 0
    development_delta_mover: int = 0
    rook_connectivity_delta_mover: int = 0
    center_control_delta_mover: int = 0
    space_advantage_delta_mover: int = 0

    # pawn structure and passed pawns
    self_created_isolated_pawn: bool = False
    self_created_doubled_pawn: bool = False
    self_created_backward_pawn: bool = False
    passed_pawn_delta_mover: int = 0
    promotion_threat_created: bool = False

    # outpost squares
    outpost_created: bool = False
    outpost_lost: bool = False
    outpost_square: Optional[str] = None
    outpost_piece: Optional[str] = None

    # other
    rooks_on_open_files_delta_mover: int = 0
    bishop_pair_delta_mover: int = 0
    tempo_gain: bool = False


# feature extractor
class FeatureExtractor:
    """
    Extract features using python-chess.
    Assumes 'before' position is legal and move is legal.
    """

    def __init__(self, engine: Optional[chess.engine.SimpleEngine] = None, time_limit: float = 0.10, multipv: int = 1):
        self.engine = engine
        self.time_limit = time_limit
        self.multipv = multipv


    def extract_move(
        self,
        before_fen: str,
        move_uci: str,
        history_uci: Optional[List[str]] = None,
        ply: Optional[int] = None,
    ) -> MoveFeatures:
        history_uci = history_uci or []

        before = chess.Board(before_fen)
        move = chess.Move.from_uci(move_uci)
        assert move in before.legal_moves, f"Illegal move {move_uci} for position {before_fen}"

        mover = before.turn

        # engine analysis
        eng = self._analyze_with_engine(before, move)

        # create after board
        after = before.copy(stack=False)
        after.push(move)

        feats = MoveFeatures(
            before_fen=before.fen(),
            after_fen=after.fen(),
            move_uci=move_uci,
            mover=side_name(mover),
            ply=ply,
        )

        # fill engine features
        self._fill_engine_fields(feats, mover, eng)

        # events
        self._fill_move_events(feats, before, after, move)

        # material / trades
        self._fill_material_and_trades(feats, before, after, move, history_uci)

        # hanging / tactical value
        self._fill_hanging_values(feats, before, after, mover)

        # king safety / open files near king
        self._fill_king_pressure(feats, before, after, mover)

        # activity / development / space / center
        self._fill_activity_development_space(feats, before, after, mover)

        # pawn structure and passed pawns and promotion threats
        self._fill_pawn_structure_and_passers(feats, before, after, mover)

        # outposts
        self._fill_outposts(feats, before, after, mover)

        # other: rooks on open files, bishop pair
        self._fill_misc_strategic(feats, before, after, mover)

        return feats


    # engine
    def _analyze_with_engine(self, before: chess.Board, played_move: chess.Move) -> MoveEngineInfo:
        if self.engine is None:
            return MoveEngineInfo(None, None, None)

        limit = chess.engine.Limit(time=self.time_limit)


        def normalize(info):
            return info[0] if isinstance(info, list) else info


        # analyse before move (best line)
        best_info = normalize(self.engine.analyse(before, limit, multipv=self.multipv))
        best_score = best_info["score"].white()
        best_pv_moves = best_info.get("pv", [])
        best_pv = [m.uci() for m in best_pv_moves]

        before_best = EngineLine(
            cp=best_score.score(mate_score=100000),
            mate=best_score.mate(),
            pv=best_pv,
        )

        # analyse after move played
        after_played_board = before.copy(stack=False)
        after_played_board.push(played_move)

        played_info = normalize(self.engine.analyse(after_played_board, limit, multipv=self.multipv))
        played_score = played_info["score"].white()
        played_pv = [m.uci() for m in played_info.get("pv", [])]

        after_played = EngineLine(
            cp=played_score.score(mate_score=100000),
            mate=played_score.mate(),
            pv=played_pv,
        )

        # analyse best move after move
        after_best = None
        if best_pv_moves:
            best_move = best_pv_moves[0]
            if best_move in before.legal_moves:
                after_best_board = before.copy(stack=False)
                after_best_board.push(best_move)

                best_after_info = normalize(self.engine.analyse(after_best_board, limit, multipv=self.multipv))
                best_after_score = best_after_info["score"].white()
                best_after_pv = [m.uci() for m in best_after_info.get("pv", [])]

                after_best = EngineLine(
                    cp=best_after_score.score(mate_score=100000),
                    mate=best_after_score.mate(),
                    pv=best_after_pv,
                )

        return MoveEngineInfo(
            before_best=before_best,
            after_best=after_best,
            after_played=after_played,
        )


    def _fill_engine_fields(self, feats: MoveFeatures, mover: chess.Color, eng: MoveEngineInfo):
        if eng.before_best is None or eng.after_played is None:
            return

        feats.eval_before_cp = eng.before_best.cp
        feats.eval_after_cp = eng.after_played.cp
        feats.mate_before = eng.before_best.mate
        feats.mate_after = eng.after_played.mate

        if feats.eval_before_cp is not None and feats.eval_after_cp is not None:
            feats.eval_swing_cp = feats.eval_after_cp - feats.eval_before_cp

        if feats.eval_before_cp is not None:
            feats.winpct_before = logistic_win_pct(feats.eval_before_cp)
        if feats.eval_after_cp is not None:
            feats.winpct_after = logistic_win_pct(feats.eval_after_cp)

        feats.pv_best = eng.before_best.pv[:10] if eng.before_best.pv else None
        feats.pv_after = eng.after_played.pv[:10] if eng.after_played.pv else None

        # compare eval after best engine move vs eval after played move
        if eng.after_best is not None and eng.after_best.cp is not None and eng.after_played.cp is not None:
            best_cp = eng.after_best.cp
            played_cp = eng.after_played.cp

            if mover == chess.WHITE:
                loss = best_cp - played_cp
            else:
                loss = played_cp - best_cp

            feats.played_vs_best_loss_cp = max(0, loss)


    # move events
    def _fill_move_events(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, move: chess.Move):
        feats.is_capture = before.is_capture(move)
        feats.is_en_passant = before.is_en_passant(move)
        feats.is_castle = before.is_castling(move)
        feats.is_promotion = move.promotion is not None

        if feats.is_promotion:
            feats.promoted_to = chess.piece_name(move.promotion)

        if feats.is_capture:
            captured = before.piece_at(move.to_square)
            if captured is None and feats.is_en_passant:
                ep_sq = move.to_square + (-8 if before.turn == chess.WHITE else 8)
                captured = before.piece_at(ep_sq)
            if captured:
                feats.captured_piece = chess.piece_name(captured.piece_type)

        feats.is_check = after.is_check()
        feats.is_mate = after.is_checkmate()

        if feats.is_check:
            king_sq = after.king(after.turn)
            if king_sq is not None:
                attackers = after.attackers(not after.turn, king_sq)
                feats.is_double_check = count_bits(attackers) >= 2


    # material and trades
    def _material_count(self, board: chess.Board) -> Dict[chess.Color, int]:
        tot = {chess.WHITE: 0, chess.BLACK: 0}
        for color in [chess.WHITE, chess.BLACK]:
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                tot[color] += PIECE_VALUE[pt] * len(board.pieces(pt, color))
        return tot


    def _piece_counts(self, board: chess.Board, color: chess.Color) -> dict[str, int]:
        return {
            "q": len(board.pieces(chess.QUEEN, color)),
            "r": len(board.pieces(chess.ROOK, color)),
            "b": len(board.pieces(chess.BISHOP, color)),
            "n": len(board.pieces(chess.KNIGHT, color)),
        }


    # helper for trade availability (mutual capture)
    def _mutual_capture_exists(self, board: chess.Board, a: chess.Color, b: chess.Color, piece_type: int) -> bool:
        a_sqs = list(board.pieces(piece_type, a))
        b_sqs = list(board.pieces(piece_type, b))
        if not a_sqs or not b_sqs:
            return False
        for sa in a_sqs:
            for sb in b_sqs:
                if board.is_attacked_by(a, sb) and board.is_attacked_by(b, sa):
                    return True
        return False


    def _fill_material_and_trades(
        self,
        feats: MoveFeatures,
        before: chess.Board,
        after: chess.Board,
        move: chess.Move,
        history_uci: List[str],
    ):
        mat_before = self._material_count(before)
        mat_after = self._material_count(after)

        mover = before.turn
        opp = opponent(mover)

        feats.material_delta_mover = (mat_after[mover] - mat_before[mover]) - (mat_after[opp] - mat_before[opp])

        def nonpawn_material(board: chess.Board) -> int:
            s = 0
            for color in [chess.WHITE, chess.BLACK]:
                for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    s += PIECE_VALUE[pt] * len(board.pieces(pt, color))
            return s

        feats.simplification_score = nonpawn_material(before) - nonpawn_material(after)

        b_m = self._piece_counts(before, mover)
        a_m = self._piece_counts(after, mover)
        b_o = self._piece_counts(before, opp)
        a_o = self._piece_counts(after, opp)

        feats.delta_queens_mover  = a_m["q"] - b_m["q"]
        feats.delta_rooks_mover   = a_m["r"] - b_m["r"]
        feats.delta_bishops_mover = a_m["b"] - b_m["b"]
        feats.delta_knights_mover = a_m["n"] - b_m["n"]

        feats.delta_queens_opp    = a_o["q"] - b_o["q"]
        feats.delta_rooks_opp     = a_o["r"] - b_o["r"]
        feats.delta_bishops_opp   = a_o["b"] - b_o["b"]
        feats.delta_knights_opp   = a_o["n"] - b_o["n"]

        feats.queen_trade  = (feats.delta_queens_mover < 0 and feats.delta_queens_opp < 0)
        feats.rook_trade   = (feats.delta_rooks_mover < 0 and feats.delta_rooks_opp < 0)
        feats.bishop_trade = (feats.delta_bishops_mover < 0 and feats.delta_bishops_opp < 0)
        feats.knight_trade = (feats.delta_knights_mover < 0 and feats.delta_knights_opp < 0)
        feats.minor_trade  = feats.bishop_trade or feats.knight_trade

        feats.bishop_for_knight_imbalance = (feats.delta_bishops_mover < 0 and feats.delta_knights_opp < 0)
        feats.knight_for_bishop_imbalance = (feats.delta_knights_mover < 0 and feats.delta_bishops_opp < 0)

        m_minor_down = (feats.delta_bishops_mover + feats.delta_knights_mover) < 0
        o_minor_down = (feats.delta_bishops_opp + feats.delta_knights_opp) < 0
        feats.exchange_trade = (
            (feats.delta_rooks_mover < 0 and o_minor_down) or
            (feats.delta_rooks_opp < 0 and m_minor_down)
        )

        feats.major_trade = False
        feats.captured_major = False
        feats.captured_minor = False
        if feats.is_capture and feats.captured_piece:
            if feats.captured_piece in ("rook", "queen"):
                feats.major_trade = True
                feats.captured_major = True
            elif feats.captured_piece in ("bishop", "knight"):
                feats.captured_minor = True


        # trade offers
        feats.queen_trade_available = self._mutual_capture_exists(after, mover, opp, chess.QUEEN)
        feats.rook_trade_available = self._mutual_capture_exists(after, mover, opp, chess.ROOK)
        feats.bishop_trade_available = self._mutual_capture_exists(after, mover, opp, chess.BISHOP)
        feats.knight_trade_available = self._mutual_capture_exists(after, mover, opp, chess.KNIGHT)

        moved_piece = before.piece_at(move.from_square)
        moved_pt = moved_piece.piece_type if moved_piece else None

        feats.queen_trade_offered = bool(moved_pt == chess.QUEEN and feats.queen_trade_available and not feats.queen_trade)
        feats.rook_trade_offered = bool(moved_pt == chess.ROOK and feats.rook_trade_available and not feats.rook_trade)
        feats.bishop_trade_offered = bool(moved_pt == chess.BISHOP and feats.bishop_trade_available and not feats.bishop_trade)
        feats.knight_trade_offered = bool(moved_pt == chess.KNIGHT and feats.knight_trade_available and not feats.knight_trade)

        feats.minor_trade_offered = feats.bishop_trade_offered or feats.knight_trade_offered
        feats.major_trade_offered = feats.queen_trade_offered or feats.rook_trade_offered


    # hanging / tactical value
    def _hanging_value(self, board: chess.Board, color: chess.Color) -> Tuple[int, Optional[str], Optional[str]]:
        """
        Returns total value + highest-value hanging piece + its square.
        Hanging = attacked more times than defended.
        """
        opp = opponent(color)
        total = 0
        highest_name = None
        highest_sq = None
        highest_val = -1

        for sq, piece in board.piece_map().items():
            if piece.color != color:
                continue
            if piece.piece_type == chess.KING:
                continue

            attackers = board.attackers(opp, sq)
            if not attackers:
                continue
            defenders = board.attackers(color, sq)

            if len(attackers) > len(defenders):
                val = PIECE_VALUE[piece.piece_type]
                total += val
                if val > highest_val:
                    highest_val = val
                    highest_name = chess.piece_name(piece.piece_type)
                    highest_sq = chess.square_name(sq)

        return total, highest_name, highest_sq


    def _fill_hanging_values(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, mover: chess.Color):
        opp = opponent(mover)

        before_self, _, _ = self._hanging_value(before, mover)
        after_self, hi_self, sq_self = self._hanging_value(after, mover)

        before_opp, _, _ = self._hanging_value(before, opp)
        after_opp, hi_opp, sq_opp = self._hanging_value(after, opp)

        feats.hanging_value_created_for_opponent = max(0, after_opp - before_opp)
        feats.hanging_value_created_for_self = max(0, after_self - before_self)

        feats.hanging_highest_piece_self = hi_self
        feats.hanging_highest_square_self = sq_self
        feats.hanging_highest_piece_opponent = hi_opp
        feats.hanging_highest_square_opponent = sq_opp

        feats.tempo_gain = feats.is_check or (feats.hanging_value_created_for_opponent >= 5)


    # king safety
    def _king_zone_squares(self, king_sq: chess.Square) -> chess.SquareSet:
        return chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])


    def _count_attacks_into_zone(self, board: chess.Board, attacker: chess.Color, zone: chess.SquareSet) -> int:
        cnt = 0
        for sq in zone:
            cnt += count_bits(board.attackers(attacker, sq))
        return cnt


    def _open_files_near_king(self, board: chess.Board, king_sq: chess.Square) -> int:
        king_file = chess.square_file(king_sq)
        files = [f for f in [king_file - 1, king_file, king_file + 1] if 0 <= f <= 7]
        open_count = 0
        for f in files:
            has_pawn = False
            for color in [chess.WHITE, chess.BLACK]:
                for sq in board.pieces(chess.PAWN, color):
                    if chess.square_file(sq) == f:
                        has_pawn = True
                        break
                if has_pawn:
                    break
            if not has_pawn:
                open_count += 1
        return open_count


    def _fill_king_pressure(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, mover: chess.Color):
        opp = opponent(mover)

        mover_king_before = before.king(mover)
        mover_king_after = after.king(mover)
        opp_king_before = before.king(opp)
        opp_king_after = after.king(opp)

        if mover_king_before is not None:
            feats.king_in_check_before = before.is_attacked_by(opp, mover_king_before)
        if mover_king_after is not None:
            feats.king_in_check_after = after.is_attacked_by(opp, mover_king_after)

        if opp_king_before is not None and opp_king_after is not None:
            zone_b = self._king_zone_squares(opp_king_before)
            zone_a = self._king_zone_squares(opp_king_after)
            atk_b = self._count_attacks_into_zone(before, mover, zone_b)
            atk_a = self._count_attacks_into_zone(after, mover, zone_a)
            feats.king_zone_attack_delta = atk_a - atk_b

            open_b = self._open_files_near_king(before, opp_king_before)
            open_a = self._open_files_near_king(after, opp_king_after)
            feats.open_files_near_enemy_king_delta = open_a - open_b

        feats.castling_rights_lost = (before.has_castling_rights(mover) and not after.has_castling_rights(mover))

        def pawn_shield(board: chess.Board, color: chess.Color) -> int:
            k = board.king(color)
            if k is None:
                return 0
            kf = chess.square_file(k)
            kr = chess.square_rank(k)
            direction = 1 if color == chess.WHITE else -1
            target_rank = kr + direction
            if not (0 <= target_rank <= 7):
                return 0
            cnt = 0
            for df in [-1, 0, 1]:
                ff = kf + df
                if 0 <= ff <= 7:
                    sq = chess.square(ff, target_rank)
                    p = board.piece_at(sq)
                    if p and p.color == color and p.piece_type == chess.PAWN:
                        cnt += 1
            return cnt

        if mover_king_before is not None and mover_king_after is not None:
            shield_b = pawn_shield(before, mover)
            shield_a = pawn_shield(after, mover)
            feats.exposed_king_delta = (shield_b - shield_a)


    # piece activity / development
    def _mobility(self, board: chess.Board, color: chess.Color) -> int:
        tmp = board.copy(stack=False)
        tmp.turn = color
        return tmp.legal_moves.count()


    def _development(self, board: chess.Board, color: chess.Color) -> int:
        dev = 0
        if color == chess.WHITE:
            start_knights = {chess.B1, chess.G1}
            start_bishops = {chess.C1, chess.F1}
        else:
            start_knights = {chess.B8, chess.G8}
            start_bishops = {chess.C8, chess.F8}

        for sq in board.pieces(chess.KNIGHT, color):
            if sq not in start_knights:
                dev += 1
        for sq in board.pieces(chess.BISHOP, color):
            if sq not in start_bishops:
                dev += 1
        return dev


    def _center_control(self, board: chess.Board, color: chess.Color) -> int:
        cnt = 0
        for sq in CENTER_SQUARES:
            cnt += count_bits(board.attackers(color, sq))
        return cnt


    def _space_advantage(self, board: chess.Board, color: chess.Color) -> int:
        total = 0
        for sq in chess.SQUARES:
            r = chess.square_rank(sq)
            if color == chess.WHITE and r >= 4:
                total += 1 if board.is_attacked_by(color, sq) else 0
            if color == chess.BLACK and r <= 3:
                total += 1 if board.is_attacked_by(color, sq) else 0
        return total


    def _fill_activity_development_space(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, mover: chess.Color):
        opp = opponent(mover)

        mob_mb = self._mobility(before, mover)
        mob_ma = self._mobility(after, mover)
        mob_ob = self._mobility(before, opp)
        mob_oa = self._mobility(after, opp)

        feats.mobility_delta_mover = mob_ma - mob_mb
        feats.mobility_delta_opponent = mob_oa - mob_ob

        dev_mb = self._development(before, mover)
        dev_ma = self._development(after, mover)
        feats.development_delta_mover = dev_ma - dev_mb

        feats.rook_connectivity_delta_mover = int(self._rooks_connected(after, mover)) - int(self._rooks_connected(before, mover))

        cc_mb = self._center_control(before, mover)
        cc_ma = self._center_control(after, mover)
        feats.center_control_delta_mover = cc_ma - cc_mb

        sp_mb = self._space_advantage(before, mover)
        sp_ma = self._space_advantage(after, mover)
        feats.space_advantage_delta_mover = sp_ma - sp_mb


    def _rooks_connected(self, board: chess.Board, color: chess.Color) -> bool:
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) != 2:
            return False
        r1, r2 = rooks
        if chess.square_rank(r1) != chess.square_rank(r2):
            return False
        rank = chess.square_rank(r1)
        f1, f2 = sorted([chess.square_file(r1), chess.square_file(r2)])
        for f in range(f1 + 1, f2):
            sq = chess.square(f, rank)
            if board.piece_at(sq) is not None:
                return False
        return True


    # pawn structure and passed pawns
    def _isolated_pawns(self, board: chess.Board, color: chess.Color) -> int:
        pawns = squares_of(board.pieces(chess.PAWN, color))
        files_with_pawns = {chess.square_file(sq) for sq in pawns}
        iso = 0
        for sq in pawns:
            f = chess.square_file(sq)
            if (f - 1 not in files_with_pawns) and (f + 1 not in files_with_pawns):
                iso += 1
        return iso


    def _doubled_pawns(self, board: chess.Board, color: chess.Color) -> int:
        pawns = squares_of(board.pieces(chess.PAWN, color))
        by_file: Dict[int, int] = {}
        for sq in pawns:
            f = chess.square_file(sq)
            by_file[f] = by_file.get(f, 0) + 1
        return sum(max(0, c - 1) for c in by_file.values())


    def _passed_pawns(self, board: chess.Board, color: chess.Color) -> int:
        opp = opponent(color)
        enemy_pawns = squares_of(board.pieces(chess.PAWN, opp))
        passed = 0
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            files = [ff for ff in [f-1, f, f+1] if 0 <= ff <= 7]
            blockers = 0
            for ep in enemy_pawns:
                ef = chess.square_file(ep)
                er = chess.square_rank(ep)
                if ef in files:
                    if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                        blockers += 1
            if blockers == 0:
                passed += 1
        return passed


    def _promotion_threat(self, board: chess.Board, color: chess.Color) -> bool:
        tmp = board.copy(stack=False)
        tmp.turn = color
        for mv in tmp.legal_moves:
            if mv.promotion is not None:
                return True
        return False


    def _fill_pawn_structure_and_passers(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, mover: chess.Color):
        iso_b = self._isolated_pawns(before, mover)
        iso_a = self._isolated_pawns(after, mover)
        dbl_b = self._doubled_pawns(before, mover)
        dbl_a = self._doubled_pawns(after, mover)

        feats.self_created_isolated_pawn = (iso_a > iso_b)
        feats.self_created_doubled_pawn = (dbl_a > dbl_b)

        pp_b = self._passed_pawns(before, mover)
        pp_a = self._passed_pawns(after, mover)
        feats.passed_pawn_delta_mover = pp_a - pp_b

        feats.promotion_threat_created = (not self._promotion_threat(before, mover)) and self._promotion_threat(after, mover)

        feats.self_created_backward_pawn = False


    # outpost squares
    def _is_outpost_square(self, board: chess.Board, color: chess.Color, sq: chess.Square) -> bool:
        piece = board.piece_at(sq)
        if piece is None or piece.color != color:
            return False
        if piece.piece_type not in (chess.KNIGHT, chess.BISHOP):
            return False

        defended_by_pawn = False
        for attacker_sq in board.attackers(color, sq):
            p = board.piece_at(attacker_sq)
            if p and p.color == color and p.piece_type == chess.PAWN:
                defended_by_pawn = True
                break
        if not defended_by_pawn:
            return False

        enemy = opponent(color)
        for attacker_sq in board.attackers(enemy, sq):
            p = board.piece_at(attacker_sq)
            if p and p.color == enemy and p.piece_type == chess.PAWN:
                return False

        return True


    def _find_outposts(self, board: chess.Board, color: chess.Color) -> Dict[chess.Square, chess.Piece]:
        out = {}
        for sq, piece in board.piece_map().items():
            if piece.color == color and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                if self._is_outpost_square(board, color, sq):
                    out[sq] = piece
        return out


    def _fill_outposts(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, mover: chess.Color):
        out_b = self._find_outposts(before, mover)
        out_a = self._find_outposts(after, mover)

        created = set(out_a.keys()) - set(out_b.keys())
        lost = set(out_b.keys()) - set(out_a.keys())

        if created:
            sq = next(iter(created))
            piece = out_a[sq]
            feats.outpost_created = True
            feats.outpost_square = chess.square_name(sq)
            feats.outpost_piece = chess.piece_name(piece.piece_type)

        if lost and not feats.outpost_created:
            sq = next(iter(lost))
            piece = out_b[sq]
            feats.outpost_lost = True
            feats.outpost_square = chess.square_name(sq)
            feats.outpost_piece = chess.piece_name(piece.piece_type)


    # other
    def _open_files(self, board: chess.Board) -> List[int]:
        open_files = []
        for f in range(8):
            has_pawn = False
            for color in [chess.WHITE, chess.BLACK]:
                for sq in board.pieces(chess.PAWN, color):
                    if chess.square_file(sq) == f:
                        has_pawn = True
                        break
                if has_pawn:
                    break
            if not has_pawn:
                open_files.append(f)
        return open_files


    def _rooks_on_open_files(self, board: chess.Board, color: chess.Color) -> int:
        open_files = set(self._open_files(board))
        cnt = 0
        for sq in board.pieces(chess.ROOK, color):
            if chess.square_file(sq) in open_files:
                cnt += 1
        return cnt


    def _bishop_pair(self, board: chess.Board, color: chess.Color) -> bool:
        return len(board.pieces(chess.BISHOP, color)) >= 2


    def _fill_misc_strategic(self, feats: MoveFeatures, before: chess.Board, after: chess.Board, mover: chess.Color):
        ro_b = self._rooks_on_open_files(before, mover)
        ro_a = self._rooks_on_open_files(after, mover)
        feats.rooks_on_open_files_delta_mover = ro_a - ro_b

        bp_b = self._bishop_pair(before, mover)
        bp_a = self._bishop_pair(after, mover)
        feats.bishop_pair_delta_mover = int(bp_a) - int(bp_b)
