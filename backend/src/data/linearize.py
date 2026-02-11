from typing import Tuple

def linearize_for_t5(data: dict) -> Tuple[str, str]:
    """
    Generates training data for Sec2Sec Model. Features
    are linearize and compacted to what is used.
    This is based purely on the templating_commentary.py.
    """
    
    # ===== ALWAYS INCLUDE (Primary decision features) =====
    features = [
        f"move: {data['move']}",
        f"stockfish_eval_delta: {data['stockfish_eval_delta']}",
        f"material_balance_delta: {data['material_balance_delta']}",
    ]
    
    # ===== ENUM FEATURES (checked in _narrate) =====
    if data.get('move_impact_delta') is not None:
        features.append(f"move_impact_delta: {data['move_impact_delta']}")
    
    if data.get('discovered_attack_or_check_delta') is not None:
        features.append(f"discovered_attack_or_check_delta: {data['discovered_attack_or_check_delta']}")
    
    # ===== PIECE ACTIVITY (checks BOTH z-score >= 0.75 AND delta) =====
    # The code does: if score >= cfg.Z_NOTICEABLE (0.75)
    piece_activity_z = data.get('piece_activity_z', 0)
    piece_activity_delta = data.get('piece_activity_delta', 0)
    if abs(piece_activity_z) >= 0.75 or piece_activity_delta != 0:
        features.append(f"piece_activity_z: {piece_activity_z}")
        features.append(f"piece_activity_delta: {piece_activity_delta}")
    
    # ===== OPEN FILES TOWARD KING (checks z >= 0.75 OR delta != 0) =====
    open_files_z = data.get('open_files_toward_king_z', 0)
    open_files_delta = data.get('open_files_toward_king_delta', 0)
    if abs(open_files_z) >= 0.75 or open_files_delta != 0:
        features.append(f"open_files_toward_king_z: {open_files_z}")
        features.append(f"open_files_toward_king_delta: {open_files_delta}")
    
    # ===== ROOKS ON OPEN FILES (checks z >= 0.75 OR delta != 0) =====
    rooks_z = data.get('rooks_on_open_files_z', 0)
    rooks_delta = data.get('rooks_on_open_files_delta', 0)
    if abs(rooks_z) >= 0.75 or rooks_delta != 0:
        features.append(f"rooks_on_open_files_z: {rooks_z}")
        features.append(f"rooks_on_open_files_delta: {rooks_delta}")
    
    # ===== PAWN STRUCTURE (checks z >= 0.75 OR delta != 0) =====
    pawn_z = data.get('pawn_structure_z', 0)
    pawn_delta = data.get('pawn_structure_delta', 0)
    if abs(pawn_z) >= 0.75 or pawn_delta != 0:
        features.append(f"pawn_structure_z: {pawn_z}")
        features.append(f"pawn_structure_delta: {pawn_delta}")
    
    # ===== TACTICAL DANGER (checks z >= 0.75 OR delta != 0) =====
    tactical_z = data.get('tactical_danger_z', 0)
    tactical_delta = data.get('tactical_danger_delta', 0)
    if abs(tactical_z) >= 0.75 or tactical_delta != 0:
        features.append(f"tactical_danger_z: {tactical_z}")
        features.append(f"tactical_danger_delta: {tactical_delta}")
    
    # ===== KING SAFETY (enum, check if not None) =====
    if data.get('king_safety_delta') is not None:
        features.append(f"king_safety_delta: {data['king_safety_delta']}")
    
    # ===== CENTER CONTROL (enum, check if not None) =====
    if data.get('center_control_delta') is not None:
        features.append(f"center_control_delta: {data['center_control_delta']}")
    
    # ===== HANGING PIECE (binary, only if == 1) =====
    if data.get('hanging_piece_delta', 0) == 1:
        features.append(f"hanging_piece_delta: 1")
    
    # ===== PROMOTION THREAT (binary, only if == 1) =====
    if data.get('promotion_threat_delta', 0) == 1:
        features.append(f"promotion_threat_delta: 1")
    
    # ===== GAME PHASE (enum, check if not None) =====
    if data.get('game_phase_delta') is not None:
        features.append(f"game_phase_delta: {data['game_phase_delta']}")
    
    # ===== WIN PERCENTAGE (checks z >= 0.75 OR abs(delta) >= 6.0) =====
    win_z = data.get('win_percentage_z', 0)
    win_delta = data.get('win_percentage_delta', 0)
    if abs(win_z) >= 0.75 or abs(win_delta) >= 6.0:
        features.append(f"win_percentage_z: {win_z}")
        features.append(f"win_percentage_delta: {win_delta}")
    
    # ===== MATE IN (checks delta != 0 OR abs(z) >= 0.75) =====
    mate_delta = data.get('mate_in_delta', 0)
    mate_z = data.get('mate_in_z', 0)
    if mate_delta != 0 or abs(mate_z) >= 0.75:
        features.append(f"mate_in_delta: {mate_delta}")
        features.append(f"mate_in_z: {mate_z}")
    
    # ===== OUTPUT =====
    x = " | ".join(features)
    y = data['comment']
    
    return x, y
