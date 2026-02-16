from __future__ import annotations
from typing import Optional, List
import random
import chess

def loss_bucket(loss_cp: Optional[int]) -> str:
    if loss_cp is None: return "NA"
    if loss_cp <= 30: return "OK"
    if loss_cp <= 80: return "INACC"
    if loss_cp <= 200: return "MIST"
    return "BLUN"

def eval_phrase(cp_bucket: str) -> str:
    return {
        "W+": "clearly better for White",
        "W=": "a bit better for White",
        "EQ": "about equal",
        "B=": "a bit better for Black",
        "B+": "clearly better for Black",
        "NA": "unclear",
    }.get(cp_bucket, "unclear")

def uci_to_san(board: chess.Board, move_uci: str) -> str:
    mv = chess.Move.from_uci(move_uci)
    return board.san(mv)

def punchy(options: List[str]) -> str:
    return random.choice(options)

class TemplateCommentator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def render(
        self,
        feats,
        before_fen: str,
        move_uci: str,
        history_uci: Optional[List[str]] = None,
        opening: Optional[object] = None,
    ) -> str:
        history_uci = history_uci or []
        before = chess.Board(before_fen)
        mover_color = before.turn
        mover = "White" if mover_color == chess.WHITE else "Black"
        opp = "Black" if mover == "White" else "White"
        san = uci_to_san(before, move_uci)

        lb = loss_bucket(getattr(feats, "played_vs_best_loss_cp", None))
        hang_self = getattr(feats, "hanging_value_created_for_self", 0)
        hang_opp = getattr(feats, "hanging_value_created_for_opponent", 0)

        hi_self = getattr(feats, "hanging_highest_piece_self", None)
        hi_opp  = getattr(feats, "hanging_highest_piece_opponent", None)

        sq_self = getattr(feats, "hanging_highest_square_self", None)
        sq_opp  = getattr(feats, "hanging_highest_square_opponent", None)


        # helpers
        def pv_hint() -> str:
            pv = getattr(feats, "pv_best", None)
            if not pv:
                return ""
            return f" Best was {self._pv_snippet(before, pv)}."


        def punch(prefixes: List[str]) -> str:
            return self.rng.choice(prefixes)


        def hang_where(piece: Optional[str], sq: Optional[str]) -> str:
            if not piece:
                return "a piece"
            if sq:
                return f"the {piece} on {sq}"
            return f"the {piece}"

        
        # forced mate
        if getattr(feats, "is_mate", False):
            return f"{punch(['That’s it!', 'Game over!'])} {mover} plays {san} — checkmate."

        if getattr(feats, "mate_after", None) is not None:
            return f"{punch(['Oh wow!', 'This is crushing!', 'That’s decisive!'])} {mover} plays {san} and a mating attack is on the board. {opp} is in huge trouble!"

        # massive blunder
        if lb in ("BLUN", "MIST"):
            if hang_self >= 5:
                target = hang_where(hi_self, sq_self)
                emo = "game-losing blunder" if lb == "BLUN" else "serious mistake"
                return (
                    f"{punch(['Oh no!', 'Disaster!'])} {mover} plays {san} and simply hangs {target}! "
                    f"This is a {emo}. {opp} can just take it."
                )

            if hang_self >= 3 and lb == "BLUN":
                target = hang_where(hi_self, sq_self)
                return (
                    f"{punch(['That’s a blunder!', 'Catastrophe!', 'Oh no!'])} {mover} goes for {san}, "
                    f"and hangs {target}. {opp} can win material immediately."
                )

            # engine/eval based blunder
            if lb == "BLUN":
                return (
                    f"{punch(['Oh no!', 'That’s a blunder!', 'Catastrophe!'])} {mover} goes for {san}, "
                    f"and it throws the game away. {opp} gets a huge chance to win here!"
                )
            else:
                return (
                    f"{punch(['That’s a mistake!', 'That\'s not a good move!'])} {mover} plays {san}, "
                    f"and it gives {opp} a big chance to push an advantage!"
                )

        # winning material
        if hang_opp >= 9:
            target = hang_where("queen", sq_opp) if (hi_opp == "queen" or hi_opp is None) else hang_where(hi_opp, sq_opp)
            return f"{punch(['What a throw!', 'What a blunder!'])} After {san}, {opp}’s {target} is hanging! {mover} is outright winning now."

        if hang_opp >= 5:
            target = hang_where(hi_opp, sq_opp)
            return f"{punch(['What a blunder!', 'A piece is hung!'])} {mover} plays {san} and {opp}’s {target} is just left hanging!"

        # checks / forcing moves
        if getattr(feats, "is_check", False):
            if getattr(feats, "is_double_check", False):
                return f"{punch(['Double check!', 'That’s brutal!'])} {mover} hits {san} with a double check, and {opp} has to try to find a way to escape the attack immediately."
            return f"{punch(['Check!', 'Forcing move!'])} {mover} plays {san}, giving a check."

        if getattr(feats, "is_promotion", False):
            promo = getattr(feats, "promoted_to", "a new piece")
            return f"{punch(['That’s huge!', 'Promotion!'])} {mover} plays {san} and promotes!"

        # trades and imbalances
        offer_line = self._trade_offer_comment(feats, mover, opp, san)
        trade_line = self._trade_comment(feats, mover, opp, san)

        if trade_line:
            return trade_line

        offered = offer_line

        # mention opening
        if history_uci and opening is not None and len(history_uci) <= 12:
            in_book = getattr(opening, "played_move_in_book", True)
            name = getattr(opening, "name", "this opening")
            eco = getattr(opening, "eco", "")
            if in_book:
                line = f"{mover} plays {san}, following theory of the {name} {f'({eco})' if eco else ''}."
            else:
                if lb == "INACC":
                    line = f"{mover} plays {san}, leaving the mainline of the {name}."
                else:
                    line = f"{mover} plays {san}, deviating from theory in the {name} {f'({eco})' if eco else ''}."
            if offered:
                line += " " + offered
            return line

        # strategy commentary
        if getattr(feats, "outpost_created", False):
            sq = getattr(feats, "outpost_square", None)
            pc = getattr(feats, "outpost_piece", None)
            line = f"{punch(['Nice idea!', 'That\'s instructive!'])} {mover} plays {san}"
            if sq and pc:
                line += f" and plants a {pc} on an outpost at {sq}."
            else:
                line += "."
            if offered:
                line += " " + offered
            return line

        if getattr(feats, "passed_pawn_delta_mover", 0) > 0:
            line = f"{punch(['Nice!', 'Good decision!'])} {mover} plays {san}, creating a passed pawn. This could be an asset that decides the endgame."
            if offered:
                line += " " + offered
            return line

        if getattr(feats, "open_files_near_enemy_king_delta", 0) > 0:
            line = f"{punch(['Pressure is building!', 'Lines are opening!', 'This could get dangerous!'])} {mover} plays {san}, opening lines toward {opp}’s king."
            if offered:
                line += " " + offered
            return line

        # inaccuracy
        if lb == "INACC":
            line = f"{punch(['Not the cleanest.', 'A bit imprecise.', 'Slight slip.'])} {mover} plays {san}."
            hint = pv_hint().strip()
            if hint:
                line += " " + hint
            if offered:
                line += " " + offered
            return line

        # neutral fallback
        base = f"{mover} plays {san}"
        if offered:
            base += f", {offered}"
        base += "."
        return f"{base} The position stays {self._eval_fallback(feats)}."


    # trade offer commentary
    def _trade_offer_comment(self, feats, mover: str, opp: str, san: str) -> Optional[str]:
        offers = []

        if getattr(feats, "queen_trade_offered", False):
            offers.append("offers a queen trade")
        if getattr(feats, "rook_trade_offered", False):
            offers.append("offers a rook trade")
        if getattr(feats, "bishop_trade_offered", False):
            offers.append("offers a bishop trade")
        if getattr(feats, "knight_trade_offered", False):
            offers.append("offers a knight trade")

        if not offers:
            return None

        if len(offers) == 1:
            return f"and {offers[0]}"
        return f"and {offers[0]} — {offers[1]}"


    def _trade_comment(self, feats, mover: str, opp: str, san: str) -> Optional[str]:
        if getattr(feats, "queen_trade", False):
            return f"{punchy(['Simplify!', 'Trading queens.', 'Into an endgame.'])} {mover} plays {san}, and the queens come off."

        if getattr(feats, "rook_trade", False):
            return f"{punchy(['Rooks get traded.', 'Heavy pieces come off.', 'Simplifying.'])} {mover} plays {san}, and the rooks are traded."

        if getattr(feats, "bishop_trade", False):
            return f"{punchy(['Bishops traded.', 'Minor pieces come off.', 'Simplification.'])} {mover} plays {san}, swapping bishops."

        if getattr(feats, "knight_trade", False):
            return f"{punchy(['Knights traded.', 'Pieces come off.', 'Simplifying.'])} {mover} plays {san}, and the knights are exchanged."

        dm_r = getattr(feats, "delta_rooks_mover", 0)
        do_r = getattr(feats, "delta_rooks_opp", 0)
        dm_b = getattr(feats, "delta_bishops_mover", 0)
        do_b = getattr(feats, "delta_bishops_opp", 0)
        dm_n = getattr(feats, "delta_knights_mover", 0)
        do_n = getattr(feats, "delta_knights_opp", 0)
        dm_q = getattr(feats, "delta_queens_mover", 0)
        do_q = getattr(feats, "delta_queens_opp", 0)

        if (dm_q < 0 and do_q < 0):
            return f"{punchy(['Trading queens.', 'Queen swap.', 'Simplify!'])} {mover} plays {san}, and the queens come off the board."
        if (dm_r < 0 and do_r < 0):
            return f"{punchy(['Rooks traded.', 'Heavy pieces off.', 'Simplification.'])} {mover} plays {san}, and the rooks are traded."
        if (dm_b < 0 and do_b < 0):
            return f"{punchy(['Bishops traded.', 'Minor-piece exchange.', 'Simplifying.'])} {mover} plays {san}, exchanging bishops."
        if (dm_n < 0 and do_n < 0):
            return f"{punchy(['Knights traded.', 'Pieces come off.', 'Simplification.'])} {mover} plays {san}, exchanging knights."

        # bishop/knight imbalance
        if getattr(feats, "bishop_for_knight_imbalance", False) or (dm_b < 0 and do_n < 0):
            return f"{punchy(['Interesting choice.', 'Imbalance!', 'Different pieces now.'])} {mover} plays {san}, giving up a bishop for a knight, creating an knight vs. bishop imbalance."
        if getattr(feats, "knight_for_bishop_imbalance", False) or (dm_n < 0 and do_b < 0):
            return f"{punchy(['Interesting trade.', 'Imbalance!', 'New structure.'])} {mover} plays {san}, exchanging a knight for a bishop, which is typically a positive trade. However, we’ll have to see who prefers this imbalance in the long-term."

        # sacrifice exchange/trade
        if getattr(feats, "exchange_trade", False):
            return f"{punchy(['Whoa!', 'Exchange!', 'Sacrifice?'])} {mover} plays {san}, and we get a rook sacrifice for a minor piece!"

        return None

    def _pv_snippet(self, board: chess.Board, pv_uci: List[str], plies: int = 2) -> str:
        tmp = board.copy(stack=False)
        out = []
        for mv_uci in pv_uci[:plies]:
            mv = chess.Move.from_uci(mv_uci)
            if mv not in tmp.legal_moves:
                break
            out.append(tmp.san(mv))
            tmp.push(mv)
        return " ".join(out) if out else "(best line)"

    def _eval_fallback(self, feats) -> str:
        cp = getattr(feats, "eval_after_cp", None)
        if cp is None:
            return "unclear"
        if cp > 150: return "better for White"
        if cp > 50:  return "slightly better for White"
        if cp >= -50: return "about equal"
        if cp >= -150: return "slightly better for Black"
        return "better for Black"
