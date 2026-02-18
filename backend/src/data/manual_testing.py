from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, List

import torch
import chess
import chess.engine
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from feature_extractor import FeatureExtractor
from template_commentator import TemplateCommentator

from modify_training_data import (
    linearize_for_t5,
    stage2_compact_record,
    stage3_positionalize,
    inject_style_token,
    normalize_punct,
)



BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "t5_chess_commentary" / "final"
DATA_PATH  = BASE_DIR / "training_data.jsonl"

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
STOCKFISH_PATH = REPO_DIR / "stockfish" / "stockfish-windows.exe"
STOCKFISH_PATH = str(STOCKFISH_PATH.resolve())

MAX_INPUT_LEN = 312
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_DO_SAMPLE = True
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_BEAMS = 1
DEFAULT_REPETITION_PENALTY = 1.1

ENGINE_TIME_LIMIT = 0.10
ENGINE_MULTIPV = 1

TEST_FEN = "r3k1r1/ppq2p2/4p2p/3pPb2/7K/2PB4/PP1NQ1P1/5R2 b q - 0 1"
TEST_MOVE = "f5d3"
STYLE = "neutral"   # neutral, hype, calm, technical, short
SHOW_TEMPLATE_OUTPUT = False



def load_model():
    # print("Loading model from:", MODEL_DIR)
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
):
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


def build_model_input_from_feats_dict(example: dict, *, style: str) -> str:
    """
    Build EXACT model input text, matching your training pipeline:
    linearize -> compact -> positionalize -> inject style
    """
    inp1, tgt1 = linearize_for_t5(example)
    inp2, tgt2 = stage2_compact_record(inp1, tgt1)
    inp3 = stage3_positionalize(inp2)
    inp4 = inject_style_token(inp3, style)
    return inp4


def fen_move_to_input(
    engine: chess.engine.SimpleEngine,
    fen: str,
    move_uci: str,
    *,
    style: str,
    history_uci: Optional[List[str]] = None,
    ply: Optional[int] = None,
) -> tuple[str, object]:
    """
    Returns (model_input_text, feats_object).
    """
    history_uci = history_uci or []

    extractor = FeatureExtractor(engine=engine, time_limit=ENGINE_TIME_LIMIT, multipv=ENGINE_MULTIPV)
    feats = extractor.extract_move(
        before_fen=fen,
        move_uci=move_uci,
        history_uci=history_uci,
        ply=ply,
    )

    example = {
        "fen": fen,
        "move": move_uci,
        "comment": "DUMMY",
    }
    example.update(feats.__dict__)

    model_input = build_model_input_from_feats_dict(example, style=style)
    return model_input, feats


def quick_manual_tests_from_fen(tok, model, engine):
    model_input, feats = fen_move_to_input(
        engine,
        TEST_FEN,
        TEST_MOVE,
        style=STYLE,
        history_uci=[],
        ply=None,
    )

    pred = generate_one(tok, model, model_input)

    print("FEN:\t", TEST_FEN)
    print("MOVE:\t", TEST_MOVE)
    # print("\nMODEL INPUT (debug):\n", model_input)
    print("\nTrained Model Output:\n", pred)

    if SHOW_TEMPLATE_OUTPUT:
        templ = TemplateCommentator(seed=0)
        template_comment = templ.render(
            feats=feats,
            before_fen=TEST_FEN,
            move_uci=TEST_MOVE,
            history_uci=[],
            opening=None,
        )
        print("\nTemplate Output:\n", template_comment)


def main():
    print("PYTHON:", __import__("sys").executable)
    print("TORCH:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("DEVICE:", DEVICE)

    tok, model = load_model()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        quick_manual_tests_from_fen(tok, model, engine)
    finally:
        try:
            engine.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
