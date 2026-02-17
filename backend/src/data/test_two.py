import json
import random
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "t5_chess_commentary" / "final"
DATA_PATH = BASE_DIR / "training_data.jsonl"

MAX_INPUT_LEN = 312

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print("Loading model from:", MODEL_DIR)
    tok = T5TokenizerFast.from_pretrained(str(MODEL_DIR))
    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    model.eval()
    return tok, model


@torch.inference_mode()
def generate_one(tok, model, text: str,
                 max_new_tokens: int = 80,
                 do_sample: bool = True,
                 temperature: float = 0.9,
                 top_p: float = 0.9,
                 num_beams: int = 1,
                 repetition_penalty: float = 1.1):
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
        early_stopping=True,
    )

    return tok.decode(out[0], skip_special_tokens=True)


def quick_manual_tests(tok, model):
    examples = [
        """task: chess_commentary
style: neutral
before_fen: 7k/p7/1bP1n1p1/4P2p/2Q2P2/2R3PK/7P/6q1 b - - 1 40
move_uci: g1b1
mover: black
features: -62 51 113 111 44.317 54.681 0 0 0 0 0 NA 0 0 0 0 knight NA 0 3 1 0 0 0 NA NA NA"""
    ]

    print("\nmanual tests")
    for i, x in enumerate(examples, 1):
        pred = generate_one(tok, model, x)
        print(f"\n--- Example {i} ---")
        print("INPUT:\n", x)
        print("\nPRED:\n", pred)


def sample_from_jsonl(tok, model, k: int = 10, seed: int = 42):
    rng = random.Random(seed)
    lines = []

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if not lines:
        raise RuntimeError("DATA_PATH appears empty.")

    picks = rng.sample(lines, k=min(k, len(lines)))

    print("\nrandom tests")
    for idx, line in enumerate(picks, 1):
        obj = json.loads(line)
        inp = obj["input"]
        tgt = obj.get("target", "")

        pred = generate_one(tok, model, inp)

        print(f"\n--- Sample {idx} ---")
        print("INPUT:\n", inp)
        if tgt:
            print("\nTARGET:\n", tgt)
        print("\nPRED:\n", pred)


def main():
    print("PYTHON:", __import__("sys").executable)
    print("TORCH:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("DEVICE:", DEVICE)

    tok, model = load_model()

    quick_manual_tests(tok, model)

    if DATA_PATH.exists():
        sample_from_jsonl(tok, model, k=5, seed=1)


if __name__ == "__main__":
    main()
