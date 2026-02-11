from state.game_state import GameState, Piece
from feature_vector import FeatureVector
from enums.board import Color, PieceType, GamePhase
from enums.eval import (
    KingSafety,
    CenterControl,
    EvaluationBucket
)
from enums.moves import MoveImpact, DiscoveredTactic

import chess
import chess.engine
from pathlib import Path
import math

class FeatureExtractor:
    @staticmethod
    def extract(state: GameState, engine) -> FeatureVector:
        features = FeatureVector()

        # --- Game state ---
        features.side_to_move = state.side_to_move
        features.game_phase = FeatureExtractor._infer_game_phase(state)

        # --- Board evaluation ---
        features.material_balance = FeatureExtractor._material_balance(state)
        features.king_safety = FeatureExtractor._king_safety(state)
        features.center_control = FeatureExtractor._center_control(state)
        features.pawn_structure = FeatureExtractor._pawn_structure(state)
        features.piece_activity = FeatureExtractor._piece_activity(state)
        features.tactical_danger = FeatureExtractor._tactical_danger(state)
        features.evaluation_bucket = FeatureExtractor._evaluation_bucket(state)

        # --- Move / Threat analysis ---
        features.move_impact = FeatureExtractor._move_impact(state)
        features.discovered_attack_or_check = FeatureExtractor._discovered_attack_or_check(state)
        features.hanging_piece = FeatureExtractor._hanging_piece(state)
        features.promotion_threat = FeatureExtractor._promotion_threat(state)
        features.open_files_toward_king = FeatureExtractor._open_files_toward_king(state)
        features.rooks_on_open_files = FeatureExtractor._rooks_on_open_files(state)

        features.sf_evaluation, features.mate_in, features.win_percentage = FeatureExtractor._evaluate_position(state, engine)

        return features


    # ------------------- FEATURE LOGIC -------------------
    @staticmethod
    def _evaluate_position(state: GameState, engine, time_limit: float = 0.02) -> float | str:
        """
        Evaluates a position using Stockfish.
        Returns a centipawn score from White's perspective,
        or "Mate in N" string if forced mate found.
        """

        # default stockfish path (uses windows stockfish in repo)

        board = chess.Board(state.fen)
        # load engine
        result = engine.analyse(
            board,
            chess.engine.Limit(time=time_limit)
        )

        score = result["score"].white()
        win_percentage = None
        if (score.score()):
            win_percentage = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * score.score())) - 1)

        return (score.score(), score.mate(), win_percentage)


    @staticmethod
    def _infer_game_phase(state: GameState) -> GamePhase:
        """Determine game phase based on piece count and types."""
        piece_count = sum(1 for row in state.grid for piece in row if piece)
        
        # Count queens - their presence indicates opening/middlegame
        queens = sum(1 for row in state.grid for piece in row 
                    if piece and piece.type == PieceType.QUEEN)
        
        if piece_count > 22 or queens == 2:
            return GamePhase.OPENING
        elif piece_count > 10:
            return GamePhase.MIDDLEGAME
        else:
            return GamePhase.ENDGAME


    @staticmethod
    def _material_balance(state: GameState) -> int:
        """Calculate material advantage (positive = white ahead)."""
        values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0
        }
        white = sum(values[piece.type] for row in state.grid for piece in row 
                   if piece and piece.color == Color.WHITE)
        black = sum(values[piece.type] for row in state.grid for piece in row 
                   if piece and piece.color == Color.BLACK)
        return white - black


    @staticmethod
    def _king_safety(state: GameState) -> KingSafety:
        """Evaluate king safety based on pawn shield and nearby pieces."""
        def evaluate_king(king_pos, color):
            if not king_pos:
                return KingSafety.EXPOSED
            
            row, col = king_pos
            enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
            
            # Check for pawn shield (pawns in front of king)
            pawn_shield_count = 0
            if color == Color.WHITE:
                # White king - check row above
                if row > 0:
                    for c in range(max(0, col-1), min(8, col+2)):
                        piece = state.grid[row-1][c]
                        if piece and piece.type == PieceType.PAWN and piece.color == color:
                            pawn_shield_count += 1
            else:
                # Black king - check row below
                if row < 7:
                    for c in range(max(0, col-1), min(8, col+2)):
                        piece = state.grid[row+1][c]
                        if piece and piece.type == PieceType.PAWN and piece.color == color:
                            pawn_shield_count += 1
            
            # Check for nearby enemy pieces (attackers)
            nearby_attackers = 0
            for r in range(max(0, row-2), min(8, row+3)):
                for c in range(max(0, col-2), min(8, col+3)):
                    piece = state.grid[r][c]
                    if piece and piece.color == enemy_color:
                        if piece.type in [PieceType.QUEEN, PieceType.ROOK, PieceType.KNIGHT]:
                            nearby_attackers += 1
            
            # Determine safety level
            if nearby_attackers >= 2:
                return KingSafety.UNDER_ATTACK
            elif pawn_shield_count >= 2 and nearby_attackers == 0:
                return KingSafety.SAFE
            else:
                return KingSafety.EXPOSED
        
        # Find kings
        white_king_pos = None
        black_king_pos = None
        for r in range(8):
            for c in range(8):
                piece = state.grid[r][c]
                if piece and piece.type == PieceType.KING:
                    if piece.color == Color.WHITE:
                        white_king_pos = (r, c)
                    else:
                        black_king_pos = (r, c)
        
        # Evaluate based on side to move
        if state.side_to_move == Color.WHITE:
            return evaluate_king(white_king_pos, Color.WHITE)
        else:
            return evaluate_king(black_king_pos, Color.BLACK)


    @staticmethod
    def _center_control(state: GameState) -> CenterControl:
        """Evaluate control of central squares."""
        center = [(3, 3), (3, 4), (4, 3), (4, 4)]
        extended_center = [(2, 2), (2, 3), (2, 4), (2, 5),
                          (3, 2), (3, 5), (4, 2), (4, 5),
                          (5, 2), (5, 3), (5, 4), (5, 5)]
        
        w = b = 0
        
        # Score center squares (worth 2 points each)
        for r, c in center:
            piece = state.grid[r][c]
            if piece:
                if piece.color == Color.WHITE:
                    w += 2
                else:
                    b += 2
        
        # Score extended center (worth 1 point each)
        for r, c in extended_center:
            piece = state.grid[r][c]
            if piece:
                if piece.color == Color.WHITE:
                    w += 1
                else:
                    b += 1
        
        if w > b + 2:
            return CenterControl.WHITE
        elif b > w + 2:
            return CenterControl.BLACK
        else:
            return CenterControl.CONTESTED


    @staticmethod
    def _pawn_structure(state: GameState):
        """Evaluate pawn structure quality."""
        def count_pawn_weaknesses():
            doubled = 0
            isolated = 0
            backward = 0
            
            # Check each file for doubled/isolated pawns
            for col in range(8):
                white_pawns_in_file = []
                black_pawns_in_file = []
                
                for row in range(8):
                    piece = state.grid[row][col]
                    if piece and piece.type == PieceType.PAWN:
                        if piece.color == Color.WHITE:
                            white_pawns_in_file.append(row)
                        else:
                            black_pawns_in_file.append(row)
                
                # Doubled pawns (2+ pawns on same file)
                if len(white_pawns_in_file) > 1:
                    doubled += len(white_pawns_in_file) - 1
                if len(black_pawns_in_file) > 1:
                    doubled += len(black_pawns_in_file) - 1
                
                # Isolated pawns (no friendly pawns on adjacent files)
                if white_pawns_in_file:
                    has_neighbor = False
                    for adj_col in [col-1, col+1]:
                        if 0 <= adj_col < 8:
                            for row in range(8):
                                piece = state.grid[row][adj_col]
                                if piece and piece.type == PieceType.PAWN and piece.color == Color.WHITE:
                                    has_neighbor = True
                                    break
                    if not has_neighbor:
                        isolated += len(white_pawns_in_file)
                
                if black_pawns_in_file:
                    has_neighbor = False
                    for adj_col in [col-1, col+1]:
                        if 0 <= adj_col < 8:
                            for row in range(8):
                                piece = state.grid[row][adj_col]
                                if piece and piece.type == PieceType.PAWN and piece.color == Color.BLACK:
                                    has_neighbor = True
                                    break
                    if not has_neighbor:
                        isolated += len(black_pawns_in_file)
            
            return doubled + isolated
        
        weaknesses = count_pawn_weaknesses()
        
        return weaknesses


    @staticmethod
    def _piece_activity(state: GameState):
        """Evaluate piece mobility and activity."""
        def count_piece_mobility(color):
            mobility = 0
            
            for r in range(8):
                for c in range(8):
                    piece = state.grid[r][c]
                    if not piece or piece.color != color:
                        continue
                    
                    # Knights: count potential squares
                    if piece.type == PieceType.KNIGHT:
                        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2),
                                       (1,-2), (1,2), (2,-1), (2,1)]
                        for dr, dc in knight_moves:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                target = state.grid[nr][nc]
                                if not target or target.color != color:
                                    mobility += 1
                    
                    # Bishops: count diagonal squares
                    elif piece.type == PieceType.BISHOP:
                        for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = r + dr, c + dc
                            while 0 <= nr < 8 and 0 <= nc < 8:
                                target = state.grid[nr][nc]
                                if not target:
                                    mobility += 1
                                    nr += dr
                                    nc += dc
                                elif target.color != color:
                                    mobility += 1
                                    break
                                else:
                                    break
                    
                    # Rooks: count horizontal/vertical squares
                    elif piece.type == PieceType.ROOK:
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            while 0 <= nr < 8 and 0 <= nc < 8:
                                target = state.grid[nr][nc]
                                if not target:
                                    mobility += 1
                                    nr += dr
                                    nc += dc
                                elif target.color != color:
                                    mobility += 1
                                    break
                                else:
                                    break
                    
                    # Queen: combination of rook + bishop
                    elif piece.type == PieceType.QUEEN:
                        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                                      (0,1), (1,-1), (1,0), (1,1)]:
                            nr, nc = r + dr, c + dc
                            while 0 <= nr < 8 and 0 <= nc < 8:
                                target = state.grid[nr][nc]
                                if not target:
                                    mobility += 1
                                    nr += dr
                                    nc += dc
                                elif target.color != color:
                                    mobility += 1
                                    break
                                else:
                                    break
            
            return mobility
        
        white_mobility = count_piece_mobility(Color.WHITE)
        black_mobility = count_piece_mobility(Color.BLACK)
        
        diff = abs(white_mobility - black_mobility)
        
        return diff


    @staticmethod
    def _tactical_danger(state: GameState):
        """Detect immediate tactical threats."""
        # Check for pieces under attack
        attacks_count = 0
        
        for r in range(8):
            for c in range(8):
                piece = state.grid[r][c]
                if not piece:
                    continue
                
                # Count attackers on this square
                attackers = FeatureExtractor._count_attackers(state, r, c, piece.color)
                if attackers > 0:
                    attacks_count += attackers
        
        return attacks_count


    @staticmethod
    def _count_attackers(state: GameState, row: int, col: int, target_color: Color) -> int:
        """Count how many enemy pieces attack a given square."""
        enemy_color = Color.BLACK if target_color == Color.WHITE else Color.WHITE
        attackers = 0
        
        # Check for enemy pawns
        if target_color == Color.WHITE:
            # White piece - check for black pawns attacking diagonally from above
            for dc in [-1, 1]:
                nr, nc = row - 1, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    piece = state.grid[nr][nc]
                    if piece and piece.type == PieceType.PAWN and piece.color == enemy_color:
                        attackers += 1
        else:
            # Black piece - check for white pawns attacking diagonally from below
            for dc in [-1, 1]:
                nr, nc = row + 1, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    piece = state.grid[nr][nc]
                    if piece and piece.type == PieceType.PAWN and piece.color == enemy_color:
                        attackers += 1
        
        # Check for knights
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
        for dr, dc in knight_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                piece = state.grid[nr][nc]
                if piece and piece.type == PieceType.KNIGHT and piece.color == enemy_color:
                    attackers += 1
        
        return attackers


    @staticmethod
    def _evaluation_bucket(state: GameState) -> EvaluationBucket:
        """Overall position evaluation."""
        material = FeatureExtractor._material_balance(state)
        
        if material >= 5:
            return EvaluationBucket.WINNING
        elif material >= 2:
            return EvaluationBucket.BETTER
        elif material <= -5:
            return EvaluationBucket.WINNING  # For opponent
        elif material <= -2:
            return EvaluationBucket.BETTER  # For opponent
        else:
            return EvaluationBucket.EQUAL


    @staticmethod
    def _move_impact(state: GameState) -> MoveImpact:
        """Estimate impact of the last move (simplified)."""
        # This would require storing previous position - using material as proxy
        material = FeatureExtractor._material_balance(state)
        
        if state.side_to_move == Color.WHITE:
            if material > 0:
                return MoveImpact.IMPROVES
            elif material < 0:
                return MoveImpact.WORSENS
        else:
            if material < 0:
                return MoveImpact.IMPROVES
            elif material > 0:
                return MoveImpact.WORSENS
        
        return MoveImpact.NEUTRAL


    @staticmethod
    def _discovered_attack_or_check(state: GameState) -> DiscoveredTactic:
        """Check for discovered attacks or checks."""
        # Find kings
        white_king_pos = None
        black_king_pos = None
        
        for r in range(8):
            for c in range(8):
                piece = state.grid[r][c]
                if piece and piece.type == PieceType.KING:
                    if piece.color == Color.WHITE:
                        white_king_pos = (r, c)
                    else:
                        black_king_pos = (r, c)
        
        # Check if either king is under attack along ranks/files/diagonals
        if white_king_pos:
            if FeatureExtractor._is_king_in_check(state, white_king_pos, Color.WHITE):
                return DiscoveredTactic.CHECK
        
        if black_king_pos:
            if FeatureExtractor._is_king_in_check(state, black_king_pos, Color.BLACK):
                return DiscoveredTactic.CHECK
        
        # Check for discovered attacks on valuable pieces
        for r in range(8):
            for c in range(8):
                piece = state.grid[r][c]
                if piece and piece.type in [PieceType.QUEEN, PieceType.ROOK]:
                    attackers = FeatureExtractor._count_attackers(state, r, c, piece.color)
                    if attackers > 0:
                        return DiscoveredTactic.ATTACK
        
        return DiscoveredTactic.NONE


    @staticmethod
    def _is_king_in_check(state: GameState, king_pos, king_color: Color) -> bool:
        """Check if a king is in check."""
        row, col = king_pos
        enemy_color = Color.BLACK if king_color == Color.WHITE else Color.WHITE
        
        # Check for enemy rooks/queens on same rank/file
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = row + dr, col + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = state.grid[nr][nc]
                if piece:
                    if piece.color == enemy_color and piece.type in [PieceType.ROOK, PieceType.QUEEN]:
                        return True
                    break
                nr += dr
                nc += dc
        
        # Check for enemy bishops/queens on diagonals
        for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = row + dr, col + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = state.grid[nr][nc]
                if piece:
                    if piece.color == enemy_color and piece.type in [PieceType.BISHOP, PieceType.QUEEN]:
                        return True
                    break
                nr += dr
                nc += dc
        
        # Check for knights
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
        for dr, dc in knight_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                piece = state.grid[nr][nc]
                if piece and piece.type == PieceType.KNIGHT and piece.color == enemy_color:
                    return True
        
        return False


    @staticmethod
    def _hanging_piece(state: GameState) -> bool:
        """Check if there are undefended pieces under attack."""
        for r in range(8):
            for c in range(8):
                piece = state.grid[r][c]
                if not piece:
                    continue
                
                attackers = FeatureExtractor._count_attackers(state, r, c, piece.color)
                if attackers > 0:
                    # Piece is attacked - check if it's defended
                    defenders = FeatureExtractor._count_defenders(state, r, c, piece.color)
                    if defenders == 0:
                        return True
        
        return False


    @staticmethod
    def _count_defenders(state: GameState, row: int, col: int, piece_color: Color) -> int:
        """Count how many friendly pieces defend a given square."""
        defenders = 0
        
        # Check for friendly pawns defending
        if piece_color == Color.WHITE:
            for dc in [-1, 1]:
                nr, nc = row + 1, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    piece = state.grid[nr][nc]
                    if piece and piece.type == PieceType.PAWN and piece.color == piece_color:
                        defenders += 1
        else:
            for dc in [-1, 1]:
                nr, nc = row - 1, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    piece = state.grid[nr][nc]
                    if piece and piece.type == PieceType.PAWN and piece.color == piece_color:
                        defenders += 1

        return defenders


    @staticmethod
    def _promotion_threat(state: GameState) -> bool:
        """
        Check if there's a pawn that can promote on the next move.
        This means:
        - White pawn on 7th rank with clear path forward
        - Black pawn on 2nd rank with clear path forward
        """
        # White pawns on 7th rank (row 1)
        for c in range(8):
            piece = state.grid[1][c]
            if piece and piece.type == PieceType.PAWN and piece.color == Color.WHITE:
                # Can move forward to promote?
                if state.grid[0][c] is None:
                    return True
                # Can capture diagonally to promote?
                if c > 0 and state.grid[0][c-1] and state.grid[0][c-1].color == Color.BLACK:
                    return True
                if c < 7 and state.grid[0][c+1] and state.grid[0][c+1].color == Color.BLACK:
                    return True
        
        # Black pawns on 2nd rank (row 6)
        for c in range(8):
            piece = state.grid[6][c]
            if piece and piece.type == PieceType.PAWN and piece.color == Color.BLACK:
                # Can move forward to promote?
                if state.grid[7][c] is None:
                    return True
                # Can capture diagonally to promote?
                if c > 0 and state.grid[7][c-1] and state.grid[7][c-1].color == Color.WHITE:
                    return True
                if c < 7 and state.grid[7][c+1] and state.grid[7][c+1].color == Color.WHITE:
                    return True
        
        return False

    @staticmethod
    def _open_files_toward_king(state: GameState) -> int:
        """Count open files near the enemy king."""
        # Find kings
        white_king_col = None
        black_king_col = None
        
        for r in range(8):
            for c in range(8):
                piece = state.grid[r][c]
                if piece and piece.type == PieceType.KING:
                    if piece.color == Color.WHITE:
                        white_king_col = c
                    else:
                        black_king_col = c
        
        def count_open_files_near(king_col):
            if king_col is None:
                return 0
            
            open_count = 0
            for col in range(max(0, king_col-1), min(8, king_col+2)):
                has_pawn = False
                for row in range(8):
                    piece = state.grid[row][col]
                    if piece and piece.type == PieceType.PAWN:
                        has_pawn = True
                        break
                if not has_pawn:
                    open_count += 1
            
            return open_count
        
        # Count for the side to move's opponent
        if state.side_to_move == Color.WHITE:
            return count_open_files_near(black_king_col)
        else:
            return count_open_files_near(white_king_col)


    @staticmethod
    def _rooks_on_open_files(state: GameState) -> int:
        """Count rooks positioned on open files."""
        open_files = set()
        
        # Find open files (no pawns)
        for col in range(8):
            has_pawn = False
            for row in range(8):
                piece = state.grid[row][col]
                if piece and piece.type == PieceType.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                open_files.add(col)
        
        # Count rooks on open files
        rooks_on_open = 0
        for col in open_files:
            for row in range(8):
                piece = state.grid[row][col]
                if piece and piece.type == PieceType.ROOK:
                    rooks_on_open += 1
        
        return rooks_on_open
    