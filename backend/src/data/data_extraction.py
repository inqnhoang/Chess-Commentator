from __future__ import annotations

from multiprocessing.util import Finalize
from multiprocessing import Pool, cpu_count
from itertools import islice
from pathlib import Path
from typing import Optional, List
import json
import sys

import chess
import chess.engine

from feature_extractor import FeatureExtractor
from template_commentator import TemplateCommentator



BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_FILE = REPO_DIR / "data" / "fen-strings.csv"

STOCKFISH_PATH = REPO_DIR / "stockfish" / "stockfish-windows.exe"
ENGINE_PATH = str(STOCKFISH_PATH.resolve())

# number of candidate moves to use per FEN
TOP_N = 5

# engine time per analyze call
TOPN_TIME_LIMIT = 0.02

# feature extractor engine time
FE_TIME_LIMIT = 0.1
FE_MULTIPV = 1

# engine resource tuning
ENGINE_HASH_MB = 32
ENGINE_THREADS = 1

# multiprocessing and resume capability
MAX_LINES = 200_000
CHUNKSIZE = 10
FLUSH_EVERY = 200

OUT_JSONL = Path(__file__).resolve().parent / "data.jsonl"
PROGRESS_FILE = Path(__file__).resolve().parent / "progress.txt"

# worker globals (one per process)
_ENGINE: Optional[chess.engine.SimpleEngine] = None
_EXTRACTOR: Optional[FeatureExtractor] = None
_COMMENTATOR: Optional[TemplateCommentator] = None


def init_worker(engine_path: str):
    """Initialize one Stockfish engine + extractor + commentator per process."""
    global _ENGINE, _EXTRACTOR, _COMMENTATOR

    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        _ENGINE.configure({"Threads": ENGINE_THREADS, "Hash": ENGINE_HASH_MB})
    except Exception:
        pass

    _EXTRACTOR = FeatureExtractor(engine=_ENGINE, time_limit=FE_TIME_LIMIT, multipv=FE_MULTIPV)
    _COMMENTATOR = TemplateCommentator(seed=0)

    Finalize(None, shutdown_worker, exitpriority=10)


def shutdown_worker():
    """close engine when worker exits (best-effort)."""
    global _ENGINE
    try:
        if _ENGINE is not None:
            _ENGINE.close()
    except Exception:
        pass


def fen_iter_resume(path: Path, limit: int, skip: int):
    with open(path, "r", encoding="utf-8", errors="replace") as infile:
        for i in range(skip):
            next(infile, None)
        for line in islice(infile, max(0, limit - skip)):
            yield line.split(",", 1)[0].strip()


def top_n_moves(fen: str, engine: chess.engine.SimpleEngine, n: int) -> List[chess.Move]:
    """Return top N engine moves (as chess.Move) from position."""
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=TOPN_TIME_LIMIT), multipv=n)
    # python-chess returns list[dict] for multipv
    moves: List[chess.Move] = []
    for entry in info:
        pv = entry.get("pv", None)
        if pv:
            moves.append(pv[0])
    return moves


def process_one_fen(fen: str):
    """Worker function: produce rows for one FEN."""
    global _ENGINE, _EXTRACTOR, _COMMENTATOR
    if _ENGINE is None or _EXTRACTOR is None or _COMMENTATOR is None:
        return []

    try:
        # get top candidate moves
        moves = top_n_moves(fen, _ENGINE, TOP_N)

        rows = []
        for mv in moves:
            move_uci = mv.uci()

            feats = _EXTRACTOR.extract_move(
                before_fen=fen,
                move_uci=move_uci,
                history_uci=None,
                ply=None,
            )

            comment = _COMMENTATOR.render(
                feats=feats,
                before_fen=fen,
                move_uci=move_uci,
                history_uci=None,
                opening=None,
            )

            row = {
                "fen": fen,
                "move": move_uci,
                "comment": comment,
            }
            row.update(feats.__dict__)

            rows.append(row)

        return rows

    except Exception:
        return []


def main():
    ep = Path(ENGINE_PATH)
    if not ep.is_file():
        raise FileNotFoundError(f"ENGINE_PATH not found: {ep.resolve()}")

    # resume capability
    start_at = 0
    if PROGRESS_FILE.exists():
        try:
            start_at = int(PROGRESS_FILE.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            start_at = 0

    processed = start_at

    num_workers = max(1, min(cpu_count() - 1, 10))

    out_f = open(OUT_JSONL, "a", encoding="utf-8")

    try:
        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(ENGINE_PATH,),
        ) as pool:
            results = pool.imap(
                process_one_fen,
                fen_iter_resume(DATA_FILE, MAX_LINES, start_at),
                chunksize=CHUNKSIZE,
            )

            for rows in results:
                processed += 1

                for row in rows:
                    out_f.write(json.dumps(row) + "\n")

                if processed % 1000 == 0:
                    print(processed)

                if processed % FLUSH_EVERY == 0:
                    out_f.flush()
                    PROGRESS_FILE.write_text(str(processed), encoding="utf-8")
    
    except KeyboardInterrupt:
        print("\nCaught Ctrl+C — terminating workers...")
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
        raise

    finally:
        out_f.flush()
        out_f.close()
        PROGRESS_FILE.write_text(str(processed), encoding="utf-8")


if __name__ == "__main__":
    main()
