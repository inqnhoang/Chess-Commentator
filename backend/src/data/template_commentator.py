from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import chess
from state.game_state import GameState

from enums.moves import MoveImpact, DiscoveredTactic
from enums.eval import KingSafety, CenterControl, EvaluationBucket
from enums.board import GamePhase



# config
@dataclass(frozen=True)
class TemplateConfig:
    # centipawn swing thresholds (delta = after - before)
    BLUNDER_CP: int = -250
    MISTAKE_CP: int = -120
    INACCURACY_CP: int = -60
    GOOD_CP: int = 80
    GREAT_CP: int = 200

    # z-score thresholds for "this feature really changed"
    Z_NOTICEABLE: float = 0.75
    Z_BIG: float = 1.5

    # pawn structure delta: weakness count (negative is improvement)
    PAWN_STRUCT_IMPROVE: int = -1
    PAWN_STRUCT_WORSEN: int = 1

    # tactical danger spike
    TACTICAL_DANGER_SPIKE: int = 3

    # max motifs to use in commentary
    MAX_MOTIFS: int = 2


class TemplateCommentator:
    """
    Commentator baseline that writes two to four sentences per move, choosing to mention the most important motifs.
    """

    def __init__(self, cfg: TemplateConfig | None = None):
        self.cfg = cfg or TemplateConfig()


    def make_comment(
        self,
        fen: str,
        move_uci: str,
        row: dict[str, Any],
        *,
        visualize: bool = False,
        visualize_next: bool = True,
    ) -> str:
        board = chess.Board(fen)
        ply = board.ply()
        actor = "White" if board.turn == chess.WHITE else "Black"
        enemy = "Black" if actor == "White" else "White"

        move = None
        san = move_uci
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                san = board.san(move)
        except Exception:
            pass

        if visualize:
            self._print_states(fen, move, visualize_next)

        return self._narrate(actor, enemy, san, row, ply)



    # visualization helper
    def _print_states(self, fen: str, move: Optional[chess.Move], visualize_next: bool) -> None:
        print("\n=== CURRENT POSITION ===")
        GameState(fen).print_board()

        if move and visualize_next:
            b = chess.Board(fen)
            if move in b.legal_moves:
                b.push(move)
                print("\n=== AFTER MOVE ===")
                GameState(b.fen()).print_board()


    # other helpers
    def _num(self, row: dict[str, Any], k: str, default: float | int = 0):
        v = row.get(k, default)
        return default if v is None else v

    def _enum(self, enum_cls, v: Any):
        if v is None:
            return None
        try:
            return enum_cls(v)
        except Exception:
            return None

    def _fmt_cp(self, cp: float | int) -> str:
        # short baseline formatting (avoid too many numbers)
        if cp >= 0:
            return f"+{int(cp)}"
        return f"{int(cp)}"


    # motif selection
    def _motifs(self, actor: str, enemy: str, row: dict[str, Any], *, exclude: set[str]) -> List[str]:
        """
        Returns up to MAX_MOTIFS motif phrases that can be woven into a paragraph.
        We score motifs by |z| or by boolean importance.
        """
        cfg = self.cfg

        def z(k): return float(self._num(row, k, 0.0))
        def n(k): return self._num(row, k, 0)

        motifs: List[Tuple[float, str, str]] = []  # (score, key, phrase)

        # activity / initiative
        if "piece_activity" not in exclude:
            score = abs(z("piece_activity_z"))
            if score >= cfg.Z_NOTICEABLE:
                phrase = f"{actor} activates their pieces" if z("piece_activity_z") > 0 else f"{actor} kind of worsens the coordination of their pieces here"
                motifs.append((score, "piece_activity", phrase))

        # open lines toward king / rook activation
        if "open_lines" not in exclude:
            score = abs(z("open_files_toward_king_z"))
            if score >= cfg.Z_NOTICEABLE or n("open_files_toward_king_delta") != 0:
                if n("open_files_toward_king_delta") > 0:
                    phrase = f"this creates an open file towards {enemy}’s king, potentially creating attacked chances in the near future"
                else:
                    phrase = f"this removes an open file towards the king, helping prevent any immediate attacks"
                motifs.append((max(score, 0.8), "open_lines", phrase))

        if "rook_file" not in exclude:
            score = abs(z("rooks_on_open_files_z"))
            if score >= cfg.Z_NOTICEABLE or n("rooks_on_open_files_delta") != 0:
                if n("rooks_on_open_files_delta") > 0:
                    phrase = f"{actor} activates a rook onto an open file"
                else:
                    phrase = f"{actor} moves a rook away from an open file, potentially letting {enemy} take control of it in the future"
                motifs.append((max(score, 0.7), "rook_file", phrase))

        # pawn structure
        if "pawn_structure" not in exclude:
            score = abs(z("pawn_structure_z"))
            if score >= cfg.Z_NOTICEABLE or n("pawn_structure_delta") != 0:
                if n("pawn_structure_delta") >= cfg.PAWN_STRUCT_WORSEN:
                    phrase = f"{actor} weakens their pawn structure, potentially leaving long-term weaknesses farther down in the game"
                elif n("pawn_structure_delta") <= cfg.PAWN_STRUCT_IMPROVE:
                    phrase = f"{actor} improves their pawn structure, eliminating potential weak pawns for the time being"
                else:
                    phrase = f"no immediate change to the pawn structure"
                motifs.append((max(score, 0.75), "pawn_structure", phrase))

        # tactics
        if "tactical" not in exclude:
            score = abs(z("tactical_danger_z"))
            if score >= cfg.Z_NOTICEABLE or n("tactical_danger_delta") != 0:
                if n("tactical_danger_delta") >= cfg.TACTICAL_DANGER_SPIKE:
                    phrase = f"the position becomes very sharp and {enemy} must be precise to not lose the game"
                elif n("tactical_danger_delta") > 0:
                    phrase = f"tactics loom in the position, and {enemy} has to stay alert to not blunder the game away"
                elif n("tactical_danger_delta") < 0:
                    phrase = f"the position simplifies, leaving no immediate tactics"
                else:
                    phrase = f"the tactical balance remains roughly unchanged"
                motifs.append((max(score, 0.7), "tactical", phrase))

        # king safety
        if "king_safety" not in exclude:
            ks = self._enum(KingSafety, row.get("king_safety_delta"))
            if ks is not None:
                if ks == KingSafety.UNDER_ATTACK:
                    motifs.append((1.2, "king_safety", f"{enemy}’s king is under heavy attack"))
                elif ks == KingSafety.EXPOSED:
                    motifs.append((1.0, "king_safety", f"{actor} weakens their own king's safety in the process"))
                else:
                    motifs.append((0.6, "king_safety", f"king safety remains fairly stable"))

        # center control
        if "center" not in exclude:
            cc = self._enum(CenterControl, row.get("center_control_delta"))
            if cc is not None:
                if cc == CenterControl.WHITE:
                    motifs.append((1.0, "center", "control of the center shifts to White"))
                elif cc == CenterControl.BLACK:
                    motifs.append((1.0, "center", "control of the center shifts to Black"))
                else:
                    motifs.append((0.8, "center", "the fight for the center becomes more contested, if one side can win the battle for the center then the advantage may tip more in their favor"))

        # hanging piece
        if "hanging" not in exclude and int(n("hanging_piece_delta")) == 1:
            motifs.append((1.3, "hanging", f"{actor} leaves something hanging, which {enemy} can just take on the next move"))

        # promotion threat
        if "promotion" not in exclude and int(n("promotion_threat_delta")) == 1:
            motifs.append((1.1, "promotion", "a pawn promotion threat appears on the board"))

        # game phase change
        if "phase" not in exclude:
            gp = self._enum(GamePhase, row.get("game_phase_delta"))
            if gp is not None:
                if gp == GamePhase.MIDDLEGAME:
                    motifs.append((0.9, "phase", "we're now entering the middlegame, where activity and king safety start to matter more. Let's see what plans each side comes up with"))
                elif gp == GamePhase.ENDGAME:
                    motifs.append((0.9, "phase", "the game heads toward an endgame, where pawn structure and passed pawns often decide the game"))
                else:
                    motifs.append((0.6, "phase", "the game is in the opening, maybe one player will outprep the other"))

        # win percentage / practical chances
        if "winp" not in exclude:
            score = abs(z("win_percentage_z"))
            wp_delta = float(self._num(row, "win_percentage_delta", 0.0))
            if score >= cfg.Z_NOTICEABLE or abs(wp_delta) >= 6.0:
                if wp_delta > 0:
                    phrase = f"the chances swing in {actor}’s favor to win the game, can they find the win"
                else:
                    phrase = f"{enemy}’s chances improve noticeably, can they convert their advantage"
                motifs.append((max(score, 0.85), "winp", phrase))

        # possible mate
        if "mate" not in exclude:
            mi = int(self._num(row, "mate_in_delta", 0))
            mz = float(self._num(row, "mate_in_z", 0.0))
            if mi != 0 or abs(mz) >= cfg.Z_NOTICEABLE:
                motifs.append((1.4, "mate", "there is a forced checkmate on the board! If it's found it's game over for {enemy}"))

        # sort by score desc, pick top MAX_MOTIFS
        motifs.sort(key=lambda x: x[0], reverse=True)

        chosen: List[str] = []
        used_keys: set[str] = set()
        for _, key, phrase in motifs:
            if key in used_keys:
                continue
            chosen.append(phrase)
            used_keys.add(key)
            if len(chosen) >= cfg.MAX_MOTIFS:
                break

        return chosen


    # def _weave_motifs(self, motifs: List[str]) -> str:
    #     """
    #     Turn zero to two motif phrases into a sentence.
    #     """
    #     if not motifs:
    #         return ""
    #     if len(motifs) == 1:
    #         return f"The point is that {motifs[0]}."
    #     # two motifs
    #     return f"The point is that {motifs[0]}. Furthermore, {motifs[1]}."

    def _cap(self, s: str) -> str:
        if not s:
            return s
        return s[0].upper() + s[1:]


    def _weave_motifs(self, motifs: List[str], ply: int) -> str:
        if not motifs:
            return ""

        def cap_first_sentence(s: str) -> str:
            return self._cap(s)

        one_templates = [
            ("bare", "{m0}."),
            ("pref", "Notably, {m0}."),
            ("pref", "A key idea here is that {m0}."),
            ("pref", "What stands out is that {m0}."),
            ("pref", "From a positional standpoint, {m0}."),
            ("pref", "Strategically speaking, {m0}."),
        ]

        two_templates = [
            ("bare", "{m0}, and {m1}."),
            ("m0_starts_sentence", "{m0}. In addition, {m1}."),
            ("m0_starts_sentence", "{m0}; this also means {m1}."),
            ("m0_starts_sentence", "{m0}. At the same time, {m1}."),
            ("pref", "First, {m0}. Second, {m1}."),
            ("m0_starts_sentence", "{m0}, which also brings {m1}."),
        ]

        if len(motifs) == 1:
            kind, tmpl = one_templates[ply % len(one_templates)]
            m0 = motifs[0]
            if kind == "bare":
                m0 = cap_first_sentence(m0)
            return tmpl.format(m0=m0)

        kind, tmpl = two_templates[ply % len(two_templates)]
        m0, m1 = motifs[0], motifs[1]

        if kind in {"bare", "m0_starts_sentence"}:
            m0 = cap_first_sentence(m0)

        if kind == "pref":
            m0 = self._cap(m0)
            m1 = self._cap(m1)

        return tmpl.format(m0=m0, m1=m1)

    
    # combining into commentary
    def _narrate(self, actor: str, enemy: str, san: str, row: dict[str, Any], ply: int) -> str:
        cfg = self.cfg

        sf = int(self._num(row, "stockfish_eval_delta", 0))
        mat = int(round(self._num(row, "material_balance_delta", 0)))
        impact = self._enum(MoveImpact, row.get("move_impact_delta"))
        tactic = self._enum(DiscoveredTactic, row.get("discovered_attack_or_check_delta"))

        # select motifs, but exclude the main theme already used
        exclude: set[str] = set()

        # tactics
        if tactic == DiscoveredTactic.CHECK:
            exclude |= {"tactical"}
            motifs = self._motifs(actor, enemy, row, exclude=exclude)
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} plays {san}, giving a check and immediately taking control of the tempo of the game. "
                f"{enemy} is forced to respond to this threat. "
                f"{support} "
                f"If {actor} follows up accurately, this forcing sequence might be converted into a lasting initiative."
            ).strip()

        if tactic == DiscoveredTactic.ATTACK:
            exclude |= {"tactical"}
            motifs = self._motifs(actor, enemy, row, exclude=exclude)
            support = self._weave_motifs(motifs, ply)
            return (
                f"With {san}, {actor} creates a discovered attack. "
                f"{enemy} suddenly has to deal with multiple threats, and may be losing material here. "
                f"{support} "
            ).strip()

        # material
        if mat >= 2:
            exclude |= {"hanging"}
            motifs = self._motifs(actor, enemy, row, exclude=exclude)
            support = self._weave_motifs(motifs, ply)
            return (
                f"{enemy} leaves something hanging, or this might be the result of a tactic, and {actor} plays {san} to win material! "
                f"If {actor} prevents {enemy} from getting counterplay and simplify, they should be able to convert this advantage. "
                f"{support} "
                f"{enemy} will to complicate the position or find some other compensation to have a chance to hold or even win the game."
            ).strip()

        if mat == 1:
            motifs = self._motifs(actor, enemy, row, exclude=set())
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} plays {san} and picks up a free pawn! "
                f"If {actor} can consolidate and create a favorable pawn structure, they could win a pawn-up endgame in the future, but this is much harder said than done. "
                f"{support} "
                f"{enemy} should look for dynamic counterplay before the extra pawn starts to matter more in the position."
            ).strip()

        if mat <= -2:
            motifs = self._motifs(actor, enemy, row, exclude=set())
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} plays {san}, but this move loses material and gives {enemy} a large, concrete advantage! "
                f"From here, {enemy} should aim to simplify and convert without taking unnecessary risks. "
                f"{support} "
                f"For {actor}, they should try to find some sort of counterplay before the material imbalance decides the game."
            ).strip()

        # big evaluation swing
        if sf <= cfg.BLUNDER_CP:
            exclude |= {"winp"}
            motifs = self._motifs(actor, enemy, row, exclude=exclude)
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} commits a major blunder with {san}! "
                f"The evaluation swings sharply in {enemy}'s favor. If {enemy} finds the correct move, the game may be decided very soon! "
                f"{support} "
                f"{enemy} should now have a straightforward route to seize control if they stay precise."
            ).strip()

        if sf <= cfg.MISTAKE_CP:
            motifs = self._motifs(actor, enemy, row, exclude=set())
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} makes a serious mistake with {san}! "
                f"If {enemy} finds the correct tactic or follow-up here, they should be in a much better position. "
                f"{support} "
                f"{enemy} should start putting pressure on here and fight for a win."
            ).strip()

        if sf <= cfg.INACCURACY_CP:
            motifs = self._motifs(actor, enemy, row, exclude=set())
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} makes an inaccuracy with {san}. "
                f"This move isn't immediately decisive, but it gives {enemy} a way to fight for a better position and potentially have winning chances. "
                f"{support} "
            ).strip()

        if sf >= cfg.GREAT_CP:
            motifs = self._motifs(actor, enemy, row, exclude=set())
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} finds an excellent move in {san}. "
                f"This keeps the initiative under their control and allows them to keep fighting for an advantage. "
                f"{support} "
            ).strip()

        if sf >= cfg.GOOD_CP:
            motifs = self._motifs(actor, enemy, row, exclude=set())
            support = self._weave_motifs(motifs, ply)
            return (
                f"{actor} plays {san}, a strong move that keeps the position in their favor. "
                f"This quietly increases pressure on the opponent and prevents them from just comfortably developing or following their own plans. "
                f"{support} "
            ).strip()

        # if no big centipawn swing, use impact + positional themes
        motifs = self._motifs(actor, enemy, row, exclude=set())
        support = self._weave_motifs(motifs, ply)

        if impact == MoveImpact.WORSENS:
            return (
                f"{actor} plays {san}, but this seems to be a step in the wrong direction. "
                f"This gives {enemy} a chance to take the initiative or improve the position without being challenged immediately. "
                f"{support} "
            ).strip()

        if impact == MoveImpact.IMPROVES:
            return (
                f"{actor} plays {san} and slightly improves their position. "
                f"{support} "
                f"From here, {actor} will want to keep improving slowly and eventually find some way to create an initiative or attack."
            ).strip()

        # neutral fallback
        return (
            f"{actor} plays {san}, keeping the position fairly equal. "
            f"{support} "
            f"If both sides continue to play good moves or keep the position equal, this game may just end in a draw."
        ).strip()
