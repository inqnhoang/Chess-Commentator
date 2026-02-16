from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data.jsonl"
OUTPUT_PATH = BASE_DIR / "training_data.jsonl"



# helpers for keeping track of progress

def count_lines_fast(path: Path) -> Optional[int]:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

class Progress:
    def __init__(self, label: str, total: Optional[int], log_every: int):
        self.label = label
        self.total = total
        self.log_every = max(1, log_every)
        self.start = time.time()

    def log(self, line_i: int, out_written: int, errors: int = 0) -> None:
        if line_i % self.log_every != 0:
            return
        elapsed = time.time() - self.start
        rate = out_written / elapsed if elapsed > 0 else 0.0

        if self.total:
            pct = (line_i / self.total) * 100.0
            lines_per_sec = line_i / elapsed if elapsed > 0 else 0.0
            remaining = self.total - line_i
            eta = remaining / lines_per_sec if lines_per_sec > 0 else 0.0
            print(
                f"[{self.label}] [{pct:6.2f}%] lines {line_i:,}/{self.total:,} | "
                f"wrote {out_written:,} | errors {errors:,} | "
                f"{rate:,.1f} out/s | elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}"
            )
        else:
            print(
                f"[{self.label}] lines {line_i:,} | wrote {out_written:,} | errors {errors:,} | "
                f"{rate:,.1f} out/s | elapsed {fmt_time(elapsed)}"
            )

    def done(self, out_written: int, errors: int) -> None:
        elapsed = time.time() - self.start
        rate = out_written / elapsed if elapsed > 0 else 0.0
        print(f"[{self.label}] Done. wrote={out_written:,} errors={errors:,} elapsed={fmt_time(elapsed)} rate={rate:,.2f}/s")



# step 1: linearize data

FEATURE_KEYS_ORDERED = [
    # engine / eval
    "eval_before_cp", "eval_after_cp", "eval_swing_cp", "played_vs_best_loss_cp",
    "mate_before", "mate_after", "winpct_before", "winpct_after",

    # move events
    "is_capture", "captured_piece", "is_check", "is_double_check", "is_mate",
    "is_promotion", "promoted_to", "is_castle", "is_en_passant",

    # trades / material / simplification
    "material_delta_mover", "simplification_score",
    "queen_trade", "rook_trade", "bishop_trade", "knight_trade",
    "minor_trade", "exchange_trade",
    "bishop_for_knight_imbalance", "knight_for_bishop_imbalance",
    "captured_major", "captured_minor",
    "queen_trade_offered", "rook_trade_offered", "bishop_trade_offered", "knight_trade_offered",
    "minor_trade_offered", "major_trade_offered",

    # hanging / tactics
    "hanging_value_created_for_self", "hanging_value_created_for_opponent",
    "hanging_highest_piece_self", "hanging_highest_square_self",
    "hanging_highest_piece_opponent", "hanging_highest_square_opponent",
    "tempo_gain",

    # king safety
    "king_in_check_before", "king_in_check_after",
    "king_zone_attack_delta", "open_files_near_enemy_king_delta",
    "castling_rights_lost", "exposed_king_delta",

    # activity / development / space
    "mobility_delta_mover", "mobility_delta_opponent",
    "development_delta_mover", "rook_connectivity_delta_mover",
    "center_control_delta_mover", "space_advantage_delta_mover",

    # pawn structure / passed pawns
    "self_created_isolated_pawn", "self_created_doubled_pawn", "self_created_backward_pawn",
    "passed_pawn_delta_mover", "promotion_threat_created",

    # outposts
    "outpost_created", "outpost_lost", "outpost_square", "outpost_piece",

    # other
    "rooks_on_open_files_delta_mover", "bishop_pair_delta_mover",
]

def _short(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        items = [str(x) for x in v[:6]]
        s = ",".join(items)
        if len(v) > 6:
            s += ",…"
        return s
    return str(v)

def linearize_for_t5(example: Dict[str, Any]) -> Tuple[str, str]:
    before_fen = example.get("before_fen") or example.get("fen") or ""
    move_uci = example.get("move_uci") or example.get("move") or ""
    mover = example.get("mover") or ""
    comment = example.get("comment") or ""

    if not before_fen or not move_uci or not comment:
        raise ValueError("Missing required fields: before_fen/fen, move_uci/move, comment")

    feat_parts: List[str] = []
    for k in FEATURE_KEYS_ORDERED:
        if k in example:
            feat_parts.append(f"{k}={_short(example[k])}")

    feats_str = " ".join(feat_parts)

    x = (
        "task: chess_commentary\n"
        f"before_fen: {before_fen}\n"
        f"move_uci: {move_uci}\n"
        f"mover: {mover}\n"
        f"features: {feats_str}"
    )
    y = comment.strip()
    return x, y



# step 2: cleanup / modify features

LINE_RE = re.compile(r"^([a-zA-Z0-9_]+):\s*(.*)$")

def parse_input_block(inp: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    lines = inp.splitlines()
    header: Dict[str, str] = {}
    features_str = ""

    for line in lines:
        if line.startswith("features:"):
            features_str = line[len("features:"):].strip()
            break
        m = LINE_RE.match(line.strip())
        if m:
            header[m.group(1)] = m.group(2)

    feats: Dict[str, str] = {}
    if features_str:
        for tok in features_str.split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            feats[k.strip()] = v.strip()

    return header, feats

def format_input_block_sorted_kv(header: Dict[str, str], feats: Dict[str, str]) -> str:
    lines: List[str] = []

    if "task" in header:
        lines.append(f"task: {header['task']}")
    else:
        lines.append("task: chess_commentary")

    for key in ["before_fen", "move_uci", "mover"]:
        if key in header:
            lines.append(f"{key}: {header[key]}")

    feature_tokens = [f"{k}={v}" for k, v in sorted(feats.items())]
    lines.append("features: " + " ".join(feature_tokens))
    return "\n".join(lines)

BEST_WAS_CLAUSE_RE = re.compile(
    r"""(?ix)
    (?:\s+|^)
    best\ was
    [^.!?]*
    (?:[.!?]\s*|$)
    """
)


def clean_target(text: str) -> str:
    if not text:
        return text
    out = BEST_WAS_CLAUSE_RE.sub(" ", text)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([.!?,;:])", r"\1", out)
    out = out.strip(" ,;")
    return out


def is_default(v: str) -> bool:
    return v in {"0", "0.0", "NA", "None", "False", "false", ""}


def collapse_trades(feats: Dict[str, str]) -> Dict[str, str]:
    def truthy(key: str) -> bool:
        v = feats.get(key)
        return v is not None and v not in {"0", "False", "false", "NA", "None", ""}

    trade_happened = "none"
    if truthy("exchange_trade"):
        trade_happened = "exchange"
    elif truthy("queen_trade"):
        trade_happened = "queens"
    elif truthy("rook_trade"):
        trade_happened = "rooks"
    elif truthy("bishop_trade"):
        trade_happened = "bishops"
    elif truthy("knight_trade"):
        trade_happened = "knights"

    trade_offered = "none"
    if truthy("queen_trade_offered"):
        trade_offered = "queens"
    elif truthy("rook_trade_offered"):
        trade_offered = "rooks"
    elif truthy("bishop_trade_offered"):
        trade_offered = "bishops"
    elif truthy("knight_trade_offered"):
        trade_offered = "knights"

    imbalance = "none"
    if truthy("bishop_for_knight_imbalance"):
        imbalance = "BforN"
    elif truthy("knight_for_bishop_imbalance"):
        imbalance = "NforB"

    drop = {
        "queen_trade", "rook_trade", "bishop_trade", "knight_trade",
        "queen_trade_offered", "rook_trade_offered", "bishop_trade_offered", "knight_trade_offered",
        "exchange_trade",
        "bishop_for_knight_imbalance", "knight_for_bishop_imbalance",
        "delta_rooks_mover", "delta_rooks_opp",
        "delta_bishops_mover", "delta_bishops_opp",
        "delta_knights_mover", "delta_knights_opp",
        "delta_queens_mover", "delta_queens_opp",
    }

    new_feats = {k: v for k, v in feats.items() if k not in drop}

    if trade_happened != "none":
        new_feats["trade_happened"] = trade_happened
    if trade_offered != "none":
        new_feats["trade_offered"] = trade_offered
    if imbalance != "none":
        new_feats["minor_imbalance"] = imbalance

    return new_feats


def stage2_compact_record(input_text: str, target_text: str) -> Tuple[str, str]:
    header, feats = parse_input_block(input_text)

    # drop square/piece fields
    feats.pop("hanging_highest_square_self", None)
    feats.pop("hanging_highest_square_opponent", None)
    feats.pop("outpost_square", None)
    feats.pop("outpost_piece", None)

    feats = collapse_trades(feats)

    feats = {k: v for k, v in feats.items() if not is_default(v)}

    new_input = format_input_block_sorted_kv(header, feats)
    new_target = clean_target(target_text)
    return new_input, new_target



# step 3: compact data further

# important for future reference! feature order: (since we dropped key=value format)
FEATURE_ORDER: List[str] = [
    "eval_before_cp",
    "eval_after_cp",
    "eval_swing_cp",
    "played_vs_best_loss_cp",
    "winpct_before",
    "winpct_after",
    "mate_after",
    "is_mate",
    "is_check",
    "is_double_check",
    "is_promotion",
    "promoted_to",
    "king_in_check_before",
    "king_in_check_after",
    "hanging_value_created_for_self",
    "hanging_value_created_for_opponent",
    "hanging_highest_piece_self",
    "hanging_highest_piece_opponent",
    "center_control_delta_mover",
    "mobility_delta_mover",
    "mobility_delta_opponent",
    "passed_pawn_delta_mover",
    "open_files_near_enemy_king_delta",
    "outpost_created",
    "trade_happened",
    "trade_offered",
    "minor_imbalance",
]

NUMERIC_DEFAULT = "0"
BOOL_DEFAULT = "0"
STR_DEFAULT = "NA"

BOOL_KEYS = {
    "is_mate",
    "is_check",
    "is_double_check",
    "is_promotion",
    "king_in_check_before",
    "king_in_check_after",
    "outpost_created",
}

STR_KEYS = {
    "promoted_to",
    "hanging_highest_piece_self",
    "hanging_highest_piece_opponent",
    "trade_happened",
    "trade_offered",
    "minor_imbalance",
}

def default_for(key: str) -> str:
    if key in STR_KEYS:
        return STR_DEFAULT
    if key in BOOL_KEYS:
        return BOOL_DEFAULT
    return NUMERIC_DEFAULT

def normalize_bool(v: str) -> str:
    if v in {"1", "0"}:
        return v
    if v.lower() in {"true", "t", "yes"}:
        return "1"
    if v.lower() in {"false", "f", "no"}:
        return "0"
    return v

def format_input_block_positional(header: Dict[str, str], feature_values: List[str]) -> str:
    lines: List[str] = []
    if "task" in header:
        lines.append(f"task: {header['task']}")
    else:
        lines.append("task: chess_commentary")

    for key in ["before_fen", "move_uci", "mover"]:
        if key in header:
            lines.append(f"{key}: {header[key]}")

    lines.append("features: " + " ".join(feature_values))
    return "\n".join(lines)

def stage3_positionalize(input_text: str) -> str:
    header, feats = parse_input_block(input_text)

    values: List[str] = []
    for k in FEATURE_ORDER:
        v = feats.get(k, default_for(k))
        if k in BOOL_KEYS:
            v = normalize_bool(v)
        values.append(v)

    return format_input_block_positional(header, values)



# step 4: add variants/paraphrasing to commentary

_whitespace_re = re.compile(r"\s+")
_space_before_punct = re.compile(r"\s+([,!.?;:])")

def normalize_punct(s: str) -> str:
    s = s.strip()
    s = _whitespace_re.sub(" ", s)
    s = _space_before_punct.sub(r"\1", s)
    s = s.replace(" .", ".")
    s = re.sub(r"\.{2,}", "...", s)
    return s

def swap_phrases(s: str, rng: random.Random) -> str:
    swaps: List[Tuple[str, List[str]]] = [
        (r"\bThat’s not a good move!\b", ["That’s a mistake!", "Not the best choice!", "That’s a bad move!"]),
        (r"\bThat’s a blunder!\b", ["Oh no—blunder!", "Huge blunder!", "That’s a disaster!"]),
        (r"\bit throws the game away\b", ["the position is coming apart", "it hands the game away", "it trhows the position away"]),
        (r"\bgives (\w+) a big chance\b", [r"hands \1 a big chance", r"gifts \1 a big opportunity", r"lets \1 take over"]),
        (r"\bThe position stays\b", ["The position remains", "It stays", "We’re still looking at"]),
        (r"\bslightly better for\b", ["a little better for", "with a small edge for", "with a slight pull for"]),
        (r"\bbetter for\b", ["favorable for", "leaning for", "with the advantage for"]),
        (r"\bout\-right winning\b", ["winning outright", "completely winning", "decisively winning"]),
    ]

    out = s
    rng.shuffle(swaps)
    num = rng.randint(1, 3)
    applied = 0
    for pattern, choices in swaps:
        if applied >= num:
            break
        if re.search(pattern, out):
            repl = rng.choice(choices)
            out = re.sub(pattern, repl, out, count=1)
            applied += 1
    return out

def vary_punctuation(s: str, rng: random.Random) -> str:
    s = s.strip()
    s = re.sub(r"^(Oh no|Huge blunder|That’s a disaster|Disaster)(!?)\s*", lambda m: m.group(1) + " — ", s)
    if "blunder" in s.lower() and s.endswith(".") and rng.random() < 0.35:
        s = s[:-1] + "!"
    return s

def shorten_if_style(s: str, style: str) -> str:
    if style != "short":
        return s
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    if len(parts) <= 1:
        return s
    out = parts[0]
    if len(parts) >= 2 and len(parts[1]) <= 60:
        out += " " + parts[1]
    return out

def apply_style(s: str, style: str, rng: random.Random) -> str:
    s = normalize_punct(s)
    s = swap_phrases(s, rng)
    s = vary_punctuation(s, rng)

    if style == "hype":
        if rng.random() < 0.4 and "!" not in s:
            s = s.replace(".", "!", 1)
    elif style == "calm":
        s = s.replace("!!", "!").replace("!", ".")
        s = normalize_punct(s)
    elif style == "technical":
        s = s.replace("big chance", "clear chance")
        s = s.replace("not a good move", "imprecision")
        s = normalize_punct(s)

    s = shorten_if_style(s, style)
    return normalize_punct(s)

def inject_style_token(inp: str, style: str) -> str:
    lines = inp.splitlines()
    if lines and lines[0].startswith("task:"):
        return "\n".join([lines[0], f"style: {style}"] + lines[1:])
    return f"style: {style}\n{inp}"



def modify_data() -> None:
    in_path = INPUT_PATH
    out_path = OUTPUT_PATH

    k_variants = 4
    add_style_token = True
    styles = ["neutral", "hype", "calm", "technical", "short"]
    seed = 42
    log_every = 50_000
    flush_every = 10_000
    max_examples = None   # set to an int for debugging

    # progress setup
    total_lines = count_lines_fast(in_path)
    prog = Progress("Modifying Data...", total_lines, log_every)

    rng = random.Random(seed)

    n_in_lines = 0
    n_out_lines = 0
    errors = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    if total_lines is not None:
        print(f"Detected ~{total_lines:,} lines in input.")
    print("Starting...\n")

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_i, line in enumerate(fin, start=1):
            if max_examples is not None and n_in_lines >= max_examples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)

                # step 1: linearize
                inp1, tgt1 = linearize_for_t5(raw)

                # step 2: cleanup + shorten / remove features
                inp2, tgt2 = stage2_compact_record(inp1, tgt1)

                # step 3: positional features
                inp3 = stage3_positionalize(inp2)

                # step 4: add variants/paraphrasing
                base_style = "neutral"
                out_inp0 = inject_style_token(inp3, base_style) if add_style_token else inp3
                out_tgt0 = normalize_punct(tgt2)
                fout.write(json.dumps({"input": out_inp0, "target": out_tgt0}, ensure_ascii=False) + "\n")
                n_out_lines += 1

                for j in range(1, k_variants):
                    style = rng.choice(styles) if add_style_token else "neutral"
                    local_rng = random.Random((n_in_lines + 1) * 1000 + j)

                    new_tgt = apply_style(tgt2, style, local_rng)
                    new_inp = inject_style_token(inp3, style) if add_style_token else inp3

                    fout.write(json.dumps({"input": new_inp, "target": new_tgt}, ensure_ascii=False) + "\n")
                    n_out_lines += 1

                n_in_lines += 1

                if n_out_lines % flush_every == 0:
                    fout.flush()

            except Exception as e:
                errors += 1
                if errors <= 5 or errors % 1000 == 0:
                    print(f"[Error] input line {line_i}: {type(e).__name__}: {e}")

            prog.log(line_i, n_out_lines, errors)

        fout.flush()

    prog.done(n_out_lines, errors)
    print(f"Input examples processed: {n_in_lines:,}")
    print(f"Output lines written:     {n_out_lines:,}")
    print(f"Wrote to: {out_path}")



if __name__ == "__main__":
    modify_data()
