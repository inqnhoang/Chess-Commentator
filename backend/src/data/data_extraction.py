'''Generate global mean and std of deltas from top 5 moves by stockfish
    RUN ONCE'''

import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import chess
import chess.engine

import numpy as np
import json

from state.game_state import GameState
from datapoint import DataPoint
from feature_extractor import FeatureExtractor

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "fen-strings.csv"
STOCKFISH_PATH = BASE_DIR / "stockfish" / "stockfish-mac"
ENGINE_PATH = str(STOCKFISH_PATH.resolve())
OUTPUT_FILE = Path(__file__).resolve().parent / "delta_mean_std.txt"

TIME_LIMIT = 0.02

numeric_delta_points = {
    "material_balance_delta": [],
    "piece_activity_delta": [],
    "pawn_structure_delta": [],
    "tactical_danger_delta": [],
    "stockfish_eval_delta": [],
    "mate_in_delta": [],
    "rooks_on_open_files_delta": [],
    "open_files_toward_king_delta": [],
    "win_percentage_delta": []
}

enum_bool_delta_points = {
    "game_phase_delta": [],
    "king_safety_delta": [],
    "center_control_delta": [],
    "evaluation_bucket_delta": [],
    "move_impact_delta": [],
    "discovered_attack_or_check_delta": [],
    "hanging_piece_delta": [],
    "promotion_threat_delta": []
}

total_data = []


def top_n_moves(fen, engine, n=5):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT), multipv=n)
    
    return [i["pv"][0] for i in info]

def state_move_variations (fen: str):
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    state = GameState(fen)
        
    moves = top_n_moves(state.fen, engine)
    
    data = []
    
    def enum_to_int(x):
        return x.value if x is not None else None

    for move in moves:
        board = chess.Board(state.fen)
        board.push(move)
        
        next_state = GameState(board.fen())
        
        data_point = DataPoint(state, move, next_state)
        deltas = data_point.compute_deltas(engine)
        
        data.append({
            "fen": state.fen,
            "side_to_move": state.side_to_move.value,
            "move": str(move),
            "phase": FeatureExtractor._infer_game_phase(next_state).value,

            "material_balance_delta": deltas.material_balance_delta,
            "piece_activity_delta": deltas.piece_activity_delta,
            "pawn_structure_delta": deltas.pawn_structure_delta,
            "tactical_danger_delta": deltas.tactical_danger_delta,
            "stockfish_eval_delta": deltas.stockfish_eval_delta,
            "mate_in_delta": deltas.mate_in_delta,
            "rooks_on_open_files_delta": deltas.rooks_on_open_files_delta,
            "open_files_toward_king_delta": deltas.open_files_toward_king_delta,
            "win_percentage_delta": deltas.win_percentage_delta,

            "game_phase_delta": enum_to_int(deltas.game_phase_delta),
            "hanging_piece_delta": 1 if deltas.hanging_piece_delta else 0,
            "promotion_threat_delta": 1 if deltas.promotion_threat_delta else 0,

            "evaluation_bucket_delta": enum_to_int(deltas.evaluation_bucket_delta),
            "move_impact_delta": enum_to_int(deltas.move_impact_delta),
            "discovered_attack_or_check_delta": enum_to_int(deltas.discovered_attack_or_check_delta),

            "king_safety_delta": enum_to_int(deltas.king_safety_delta),
            "center_control_delta": enum_to_int(deltas.center_control_delta),
        })

    engine.close()
    
    return data

def main ():
    num_workers = cpu_count() - 1
    if num_workers < 1:
        num_workers = 1

    with open(DATA_FILE, 'r') as infile:
        fen_strings = [line.split(',')[0] for line in infile]

    counter = 0
    with Pool(processes=num_workers) as pool:
        results = pool.imap(state_move_variations, fen_strings)

        for i, move_deltas_list in enumerate(results, 1):
            total_data.extend(move_deltas_list) 
            for deltas in move_deltas_list:
                for key in numeric_delta_points:
                    numeric_delta_points[key].append(deltas[key])
                
                for key in enum_bool_delta_points:
                    enum_bool_delta_points[key].append(deltas[key])
            
            counter += 1
            if counter % 1000 == 0:
                break
            print(counter)
        
    with open(OUTPUT_FILE, "w+") as outfile:
        for key, deltas in numeric_delta_points.items():
            if len(deltas) == 0:
                continue
            
            arr = np.array(deltas, dtype=float)
            mean = np.mean(arr)
            std = np.std(arr)

            outfile.write(f"{key}: {mean:.6f}, {std:.6f}\n")
    
    with open("data.json", "w") as f:
        json.dump(total_data, f, indent=2)

if __name__ == "__main__":
    main()