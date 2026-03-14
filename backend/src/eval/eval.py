# backend/src/eval/eval.py
from __future__ import annotations

import time
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import chess
import chess.engine
from transformers import T5ForConditionalGeneration, T5TokenizerFast

import gemini
import openAI


# =============================================================================
# CHANGES (minimal):
# - Use fen-strings.csv (fen,eval) instead of data.jsonl
# - Skip first 200,000 lines (start at 200,001st FEN)
# - For each sampled FEN, pick a move with a mix of best/good/mistake/blunder
# =============================================================================

EVAL_DIR = Path(__file__).resolve().parent
SRC_DIR = EVAL_DIR.parent
DATA_DIR = SRC_DIR / "data"
BACKEND_DIR = SRC_DIR.parent

# Make data/ importable (feature_extractor, modify_training_data)
sys.path.append(str(DATA_DIR.resolve()))

from feature_extractor import FeatureExtractor
from modify_training_data import (
    linearize_for_t5,
    stage2_compact_record,
    stage3_positionalize,
    normalize_punct,
)

# Output dir (unchanged except name you set)
OUT_DIR = EVAL_DIR / "out_1k_modprompt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = DATA_DIR / "t5_chess_commentary_v2" / "final"

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
STOCKFISH_PATH = REPO_DIR / "stockfish" / "stockfish-windows.exe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_LEN = 312

DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_DO_SAMPLE = True
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_BEAMS = 1
DEFAULT_REPETITION_PENALTY = 1.1

ENGINE_TIME_LIMIT = 0.10
ENGINE_MULTIPV = 1

FEN_CSV = str(REPO_DIR / "data" / "fen-strings.csv")
FEN_START_INDEX = 200_000
SAMPLE_N = 1000
SEED = 42

MOVE_POLICY = [
    ("best", 0.50),
    ("good", 0.45),
    ("mistake", 0.038),
    ("blunder", 0.012),
]
TOP_N = 5
TOPN_TIME_LIMIT = 0.02
MOVE_EVAL_TIME = 0.01
MISTAKE_DROP_CP = 120
BLUNDER_DROP_CP = 300

RUBRIC_CATEGORIES = [
    "faithfulness",
    "relevance",
    "informativeness",
    "clarity",
    "human_likeness",
    "conciseness",
]
PREFERENCE_VALUES = {"native", "outsource", "tie"}


@dataclass(frozen=True)
class EvalPoint:
    fen_before: str
    move_uci: str
    fen_after: str
    source: str


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_after_fen(before_fen: str, move_uci: str) -> str:
    board = chess.Board(before_fen)
    mv = chess.Move.from_uci(move_uci)
    if mv not in board.legal_moves:
        raise ValueError(f"Illegal move {move_uci} in position {before_fen}")
    board.push(mv)
    return board.fen()


def fen_iter_from_csv(path: Path, *, start_index: int) -> Iterable[str]:
    """
    File lines look like:
        <fen>,<eval>
    We take the part before the first comma and skip the first start_index lines.
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(max(0, start_index)):
            if f.readline() == "":
                return

        for line in f:
            line = line.strip()
            if not line:
                continue
            fen = line.split(",", 1)[0].strip()
            if fen:
                yield fen


def score_to_cp(score: chess.engine.PovScore) -> int:
    """
    Convert engine score to centipawns from side-to-move POV.
    Treat mate scores as huge cp so we can compute drops.
    """
    s = score.pov(score.turn)
    if s.is_mate():
        mate_in = s.mate()
        return 100000 if (mate_in is not None and mate_in > 0) else -100000
    return int(s.score(mate_score=100000) or 0)


def top_n_moves_and_best_cp(board: chess.Board, engine: chess.engine.SimpleEngine, n: int) -> Tuple[List[chess.Move], int]:
    info = engine.analyse(board, chess.engine.Limit(time=TOPN_TIME_LIMIT), multipv=n)
    moves: List[chess.Move] = []
    best_cp: Optional[int] = None

    for entry in info:
        pv = entry.get("pv")
        if pv:
            mv = pv[0]
            moves.append(mv)
            if best_cp is None and entry.get("score") is not None:
                best_cp = score_to_cp(entry["score"])

    if best_cp is None:
        entry = engine.analyse(board, chess.engine.Limit(time=TOPN_TIME_LIMIT))
        best_cp = score_to_cp(entry["score"])

    return moves, best_cp


def eval_move_cp(board: chess.Board, move: chess.Move, engine: chess.engine.SimpleEngine) -> int:
    b2 = board.copy(stack=False)
    b2.push(move)
    entry = engine.analyse(b2, chess.engine.Limit(time=MOVE_EVAL_TIME))
    sc = entry.get("score")
    if sc is None:
        return 0
    return score_to_cp(sc)


def choose_policy(rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for name, p in MOVE_POLICY:
        acc += p
        if r <= acc:
            return name
    return MOVE_POLICY[-1][0]


def pick_move_for_fen(fen: str, engine: chess.engine.SimpleEngine, rng: random.Random) -> Optional[str]:
    board = chess.Board(fen)
    if board.is_game_over():
        return None

    top_moves, best_cp = top_n_moves_and_best_cp(board, engine, TOP_N)
    if not top_moves:
        return None

    policy = choose_policy(rng)

    if policy == "best":
        return top_moves[0].uci()

    if policy == "good":
        choices = top_moves[1:] if len(top_moves) > 1 else top_moves
        return rng.choice(choices).uci()

    threshold = BLUNDER_DROP_CP if policy == "blunder" else MISTAKE_DROP_CP

    legal = list(board.legal_moves)
    rng.shuffle(legal)

    top_set = {m.uci() for m in top_moves}
    candidates = [m for m in legal if m.uci() not in top_set] + [m for m in legal if m.uci() in top_set]

    for mv in candidates[:5]:
        mv_cp = eval_move_cp(board, mv, engine)
        drop = best_cp - mv_cp
        if drop >= threshold:
            return mv.uci()

    return rng.choice(legal).uci()


def extract_eval_points_from_fens(
    fen_csv: str,
    *,
    sample_n: int,
    seed: int,
    start_index: int,
    engine: chess.engine.SimpleEngine,
) -> List[EvalPoint]:
    """
    Sample positions from fen_csv after skipping start_index.
    For each sampled FEN, generate ONE move (best/good/mistake/blunder).
    Uses reservoir sampling so we don't load the whole file into memory.
    """
    rng = random.Random(seed)
    path = Path(fen_csv)

    reservoir: List[Tuple[str, str, str, str]] = []
    seen_valid = 0

    for fen in fen_iter_from_csv(path, start_index=start_index):
        try:
            mv_uci = pick_move_for_fen(fen, engine, rng)
            if mv_uci is None:
                continue
            fen_after = compute_after_fen(fen, mv_uci)
            reservoir.append((fen, mv_uci, fen_after, "fen_csv"))
            if len(reservoir) >= sample_n:
                break  # ✅ STOP EARLY
        except Exception:
            continue

    points = [EvalPoint(f, m, a, s) for (f, m, a, s) in reservoir]
    if not points:
        raise ValueError("No valid EvalPoints sampled from fen CSV (after skipping).")
    return points


def load_native_model() -> Tuple[T5TokenizerFast, T5ForConditionalGeneration]:
    tok = T5TokenizerFast.from_pretrained(str(MODEL_DIR))
    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    model.eval()
    return tok, model


@torch.inference_mode()
def generate_one(
    tok: T5TokenizerFast,
    model: T5ForConditionalGeneration,
    text: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    num_beams: int = DEFAULT_NUM_BEAMS,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> str:
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(DEVICE)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
    )

    return normalize_punct(tok.decode(out[0], skip_special_tokens=True))


def build_model_input_from_feats_dict(example: Dict[str, Any]) -> str:
    inp1, tgt1 = linearize_for_t5(example)
    inp2, tgt2 = stage2_compact_record(inp1, tgt1)
    inp3 = stage3_positionalize(inp2)
    return inp3


def native_commentary_for_fen_move(
    tok: T5TokenizerFast,
    model: T5ForConditionalGeneration,
    engine: chess.engine.SimpleEngine,
    *,
    fen_before: str,
    move_uci: str,
    history_uci: Optional[List[str]] = None,
    ply: Optional[int] = None,
) -> str:
    history_uci = history_uci or []
    extractor = FeatureExtractor(engine=engine, time_limit=ENGINE_TIME_LIMIT, multipv=ENGINE_MULTIPV)

    feats = extractor.extract_move(
        before_fen=fen_before,
        move_uci=move_uci,
        history_uci=history_uci,
        ply=ply,
    )

    example: Dict[str, Any] = {"fen": fen_before, "move": move_uci, "comment": "DUMMY"}
    example.update(feats.__dict__)

    model_input = build_model_input_from_feats_dict(example)
    return generate_one(tok, model, model_input)


def prompt_outsource_commentary(fen_before: str, move_uci: str, fen_after: str) -> str:
    out = openAI.prompt(fen_before, move_uci, fen_after)
    if not isinstance(out, str):
        raise TypeError(f"openAI.prompt(...) must return str, got {type(out)}")
    return out.strip()


def prompt_model_eval(
    fen_before: str,
    move_uci: str,
    fen_after: str,
    native_output: str,
    outsource_output: str,
) -> Dict[str, Any]:
    out = gemini.prompt(fen_before, move_uci, fen_after, native_output, outsource_output)

    if isinstance(out, dict):
        return out

    if isinstance(out, str):
        s = out.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"raw": s}

    return {"raw": str(out)}


def flatten_eval_row(row: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "fen_before": row.get("fen_before"),
        "move_uci": row.get("move_uci"),
        "fen_after": row.get("fen_after"),
        "source": row.get("source"),
        "native_output": row.get("native_output"),
        "outsource_output": row.get("outsource_output"),
    }

    ev = row.get("eval")
    if isinstance(ev, dict):
        if "raw" in ev:
            flat["eval_raw"] = ev.get("raw")

        pref = ev.get("preference")
        if pref is not None:
            flat["preference"] = pref

        notes = ev.get("notes")
        if notes is not None:
            flat["notes"] = notes

        native_scores = ev.get("native")
        outsource_scores = ev.get("outsource")
        if isinstance(native_scores, dict):
            for k, v in native_scores.items():
                flat[f"native_{k}"] = v
        if isinstance(outsource_scores, dict):
            for k, v in outsource_scores.items():
                flat[f"outsource_{k}"] = v
    else:
        flat["eval_raw"] = str(ev)

    return flat


def compute_stats(flat_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pref_counts = {"native": 0, "outsource": 0, "tie": 0, "missing": 0}

    hist: Dict[str, Dict[str, Dict[int, int]]] = {
        "native": {c: {i: 0 for i in range(1, 6)} for c in RUBRIC_CATEGORIES},
        "outsource": {c: {i: 0 for i in range(1, 6)} for c in RUBRIC_CATEGORIES},
    }

    sums = {"native": {c: 0 for c in RUBRIC_CATEGORIES}, "outsource": {c: 0 for c in RUBRIC_CATEGORIES}}
    counts = {"native": {c: 0 for c in RUBRIC_CATEGORIES}, "outsource": {c: 0 for c in RUBRIC_CATEGORIES}}

    for r in flat_rows:
        pref = r.get("preference")
        if pref in PREFERENCE_VALUES:
            pref_counts[str(pref)] += 1
        else:
            pref_counts["missing"] += 1

        for c in RUBRIC_CATEGORIES:
            n = r.get(f"native_{c}")
            o = r.get(f"outsource_{c}")
            if isinstance(n, int) and 1 <= n <= 5:
                hist["native"][c][n] += 1
                sums["native"][c] += n
                counts["native"][c] += 1
            if isinstance(o, int) and 1 <= o <= 5:
                hist["outsource"][c][o] += 1
                sums["outsource"][c] += o
                counts["outsource"][c] += 1

    total = sum(pref_counts.values()) or 1
    pref_pct = {k: v / total for k, v in pref_counts.items()}

    avgs = {
        side: {c: (sums[side][c] / counts[side][c]) if counts[side][c] else None for c in RUBRIC_CATEGORIES}
        for side in ("native", "outsource")
    }

    return {
        "n": len(flat_rows),
        "preference_counts": pref_counts,
        "preference_pct": pref_pct,
        "rubric_averages": avgs,
        "histograms": hist,
    }


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    """
    Returns set of (fen_before, move_uci) already present in existing jsonl.
    If file is missing, returns empty set.
    """
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                fen = obj.get("fen_before")
                mv = obj.get("move_uci")
                if isinstance(fen, str) and isinstance(mv, str):
                    done.add((fen, mv))
            except Exception:
                continue
    return done


def run_eval(
    *,
    sample_n: int,
    seed: int,
    resume: bool = True,
) -> None:
    tok, model = load_native_model()
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH.resolve()))

    points = extract_eval_points_from_fens(
        FEN_CSV,
        sample_n=sample_n,
        seed=seed,
        start_index=FEN_START_INDEX,
        engine=engine,
    )

    out_jsonl = OUT_DIR / "eval_results.jsonl"
    out_csv = OUT_DIR / "eval_results.csv"
    out_stats = OUT_DIR / "eval_stats.json"

    done_keys = load_completed_keys(out_jsonl) if resume else set()
    if done_keys:
        print(f"[resume] found {len(done_keys)} completed rows in {out_jsonl}")

    mode = "a" if (resume and out_jsonl.exists()) else "w"

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open(mode, encoding="utf-8") as fjsonl:
        try:
            written_now = 0
            for i, dp in enumerate(points, start=1):
                key = (dp.fen_before, dp.move_uci)
                if resume and key in done_keys:
                    continue

                t1 = time.time()

                try:
                    outsource_output = prompt_outsource_commentary(dp.fen_before, dp.move_uci, dp.fen_after)
                except Exception as e:
                    outsource_output = f"(openai_failed: {type(e).__name__}: {e})"

                try:
                    native_output = native_commentary_for_fen_move(
                        tok,
                        model,
                        engine,
                        fen_before=dp.fen_before,
                        move_uci=dp.move_uci,
                        history_uci=[],
                        ply=None,
                    )
                except Exception as e:
                    native_output = f"(native_failed: {type(e).__name__}: {e})"

                try:
                    ev = prompt_model_eval(dp.fen_before, dp.move_uci, dp.fen_after, native_output, outsource_output)
                except Exception as e:
                    ev = {"raw": None, "error": f"judge_failed: {type(e).__name__}: {e}"}

                row = {
                    "fen_before": dp.fen_before,
                    "move_uci": dp.move_uci,
                    "fen_after": dp.fen_after,
                    "source": dp.source,
                    "native_output": native_output,
                    "outsource_output": outsource_output,
                    "eval": ev,
                }

                rows.append(row)
                fjsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_now += 1

                if written_now % 10 == 0:
                    fjsonl.flush()

                if written_now % 5 == 0:
                    dt = time.time() - t0
                    avg = dt / max(1, written_now)
                    one = time.time() - t1
                    print(f"[eval] wrote {written_now} new rows | last={one:.2f}s avg_new={avg:.2f}s")

        finally:
            try:
                engine.close()
            except Exception:
                pass

    all_rows: List[Dict[str, Any]] = []
    with out_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    all_rows.append(obj)
            except Exception:
                continue

    flat_rows = [flatten_eval_row(r) for r in all_rows]
    write_csv(out_csv, flat_rows)

    stats = compute_stats(flat_rows)
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nWrote:\n- {out_jsonl}\n- {out_csv}\n- {out_stats}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    print("MODEL_DIR:", MODEL_DIR)
    print("STOCKFISH_PATH:", STOCKFISH_PATH)
    print("FEN_CSV:", FEN_CSV)
    print("FEN_START_INDEX:", FEN_START_INDEX)
    print("SAMPLE_N:", SAMPLE_N, "SEED:", SEED)
    print("MOVE_POLICY:", MOVE_POLICY)
    print("TOP_N:", TOP_N, "TOPN_TIME_LIMIT:", TOPN_TIME_LIMIT)
    print("MISTAKE_DROP_CP:", MISTAKE_DROP_CP, "BLUNDER_DROP_CP:", BLUNDER_DROP_CP)

    run_eval(sample_n=SAMPLE_N, seed=SEED, resume=True)