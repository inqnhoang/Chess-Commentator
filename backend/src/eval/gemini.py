"""
LLM as a Judge (Gemini)
Input Tokens:  $0.075 / 1M
Output Tokens: $0.30  / 1M
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment (.env).")

genai.configure(api_key=google_key)
model = genai.GenerativeModel("gemini-2.5-flash")

CATS = ["faithfulness", "relevance", "informativeness", "clarity", "human_likeness", "conciseness"]
PREFS = {"A", "B", "TIE"}

JUDGE_MAX_RETRIES = 4
JUDGE_RETRY_BACKOFF_S = 2.0


def _judge_prompt(
    *,
    fen_before: str,
    move: str,
    fen_after: str,
    commentary_a: str,
    commentary_b: str,
) -> str:
    return f"""
You are grading two chess commentary candidates (A vs B) for the SAME move.

POSITION CONTEXT:
Before FEN: {fen_before}
Move (UCI): {move}
After FEN: {fen_after}

CANDIDATE A:
{commentary_a}

CANDIDATE B:
{commentary_b}

CRITERIA (score each 1–5 for A and for B):
1) faithfulness (accuracy / non-hallucination)
2) relevance (focus on move and salient consequences)
3) informativeness (specific, useful analysis without inventing facts)
4) clarity (readability)
5) human_likeness (natural commentary style)
6) conciseness (no rambling)

HARD RULE (faithfulness=1):
If candidate claims check/checkmate/capture/promotion/trade/hanging-piece win OR a forced tactic WITHOUT clear support from the FEN+move itself, set faithfulness=1.

AGGREGATION:
total = sum of the 6 criteria.
winner = higher total, ties broken by higher faithfulness, then relevance, then informativeness, else tie.

OUTPUT:
Return ONLY valid JSON. No markdown. No extra text.

Schema:
{{
  "winner": "A" | "B" | "TIE",
  "A": {{
    "faithfulness": int, "relevance": int, "informativeness": int,
    "clarity": int, "human_likeness": int, "conciseness": int,
    "total": int
  }},
  "B": {{
    "faithfulness": int, "relevance": int, "informativeness": int,
    "clarity": int, "human_likeness": int, "conciseness": int,
    "total": int
  }},
  "notes": "short reason (1-2 sentences)"
}}

Constraints:
- Each criterion must be an integer 1..5.
- total must equal the sum of the 6 criteria.
""".strip()


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if lines and lines[0].lower().startswith("json"):
            s = "\n".join(lines[1:]).strip()
        else:
            s = "\n".join(lines).strip()
    return s


def _best_effort_json_object(text: str) -> str:
    """
    If model returns extra text, try to extract the first top-level {...} block.
    This is best-effort and avoids crashing long runs.
    """
    s = _strip_code_fences(text)
    if s.startswith("{") and s.endswith("}"):
        return s

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()
    return s.strip()


def _parse_and_validate_json(text: str) -> Dict[str, Any]:
    raw = text
    s = _best_effort_json_object(text)

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return {"_error": f"JSONDecodeError: {e}", "raw": raw}

    if not isinstance(obj, dict):
        return {"_error": "Judge output JSON is not an object.", "raw": raw}

    winner = obj.get("winner")
    if winner not in PREFS:
        obj["_error"] = f"Invalid winner: {winner}"
        obj["raw"] = raw
        return obj

    for side in ("A", "B"):
        block = obj.get(side)
        if not isinstance(block, dict):
            obj["_error"] = f"Missing or invalid block: {side}"
            obj["raw"] = raw
            return obj

        # coerce scores
        score_sum = 0
        for c in CATS:
            v = block.get(c)
            if not isinstance(v, int):
                try:
                    v = int(v)
                except Exception:
                    v = 1
            v = max(1, min(5, v))
            block[c] = v
            score_sum += v

        total = block.get("total")
        if not isinstance(total, int):
            total = score_sum
        block["total"] = score_sum

    notes = obj.get("notes")
    if notes is None:
        obj["notes"] = ""
    elif not isinstance(notes, str):
        obj["notes"] = str(notes)

    return obj


def _call_judge_once(prompt: str) -> str:
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def _call_judge(prompt: str) -> Dict[str, Any]:
    last_text = ""
    last_err: Optional[str] = None

    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            text = _call_judge_once(prompt)
            last_text = text
            if not text:
                last_err = "Gemini returned empty response."
                raise RuntimeError(last_err)

            obj = _parse_and_validate_json(text)
            if "_error" not in obj:
                return obj

            last_err = obj.get("_error", "unknown parse error")
            raise RuntimeError(last_err)

        except Exception as e:
            if attempt < JUDGE_MAX_RETRIES:
                time.sleep(JUDGE_RETRY_BACKOFF_S * attempt)
            else:
                return {
                    "_error": f"judge_failed: {type(e).__name__}: {e}",
                    "raw": last_text,
                }


def prompt(
    fen_before: str,
    move: str,
    fen_after: str,
    native_output: str,
    ai_output: str,
) -> Dict[str, Any]:
    """
    Returns:
    {
      "preference": "native" | "outsource" | "tie",
      "native": { six criteria 1..5, "total": int },
      "outsource": { ... },
      "notes": str,
      "raw_passes": { "pass1": <judge json>, "pass2": <judge json> },
      (optional) "error": str
    }
    """

    # pass 1: A=native, B=outsourced
    p1 = _judge_prompt(
        fen_before=fen_before,
        move=move,
        fen_after=fen_after,
        commentary_a=native_output,
        commentary_b=ai_output,
    )
    j1 = _call_judge(p1)

    p2 = _judge_prompt(
        fen_before=fen_before,
        move=move,
        fen_after=fen_after,
        commentary_a=ai_output,
        commentary_b=native_output,
    )
    j2 = _call_judge(p2)

    if "_error" in j1 or "_error" in j2:
        err = " | ".join([str(j1.get("_error", "")), str(j2.get("_error", ""))]).strip(" |")
        return {
            "preference": "tie",
            "native": {c: 3 for c in CATS} | {"total": 18},
            "outsource": {c: 3 for c in CATS} | {"total": 18},
            "notes": "",
            "raw_passes": {"pass1": j1, "pass2": j2},
            "error": err,
        }

    n1 = dict(j1["A"])
    o1 = dict(j1["B"])

    o2 = dict(j2["A"])
    n2 = dict(j2["B"])

    def avg_score(a: int, b: int) -> int:
        x = int(round((a + b) / 2.0))
        return max(1, min(5, x))

    native: Dict[str, Any] = {}
    outsource: Dict[str, Any] = {}
    for c in CATS:
        native[c] = avg_score(int(n1[c]), int(n2[c]))
        outsource[c] = avg_score(int(o1[c]), int(o2[c]))

    native["total"] = sum(native[c] for c in CATS)
    outsource["total"] = sum(outsource[c] for c in CATS)

    def tiebreak(n: Dict[str, Any], o: Dict[str, Any]) -> str:
        if n["total"] > o["total"]:
            return "native"
        if o["total"] > n["total"]:
            return "outsource"

        if n["faithfulness"] != o["faithfulness"]:
            return "native" if n["faithfulness"] > o["faithfulness"] else "outsource"
        if n["relevance"] != o["relevance"]:
            return "native" if n["relevance"] > o["relevance"] else "outsource"
        if n["informativeness"] != o["informativeness"]:
            return "native" if n["informativeness"] > o["informativeness"] else "outsource"
        return "tie"

    preference = tiebreak(native, outsource)
    notes = " | ".join([j1.get("notes", ""), j2.get("notes", "")]).strip(" |")

    return {
        "preference": preference,
        "native": native,
        "outsource": outsource,
        "notes": notes,
        "raw_passes": {"pass1": j1, "pass2": j2},
    }