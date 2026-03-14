''' Generate AI commentary
    Input Cost: $0.15 / 1M tokens ($0.00015 per 1K)
    Output Cost: $0.60 / 1M tokens ($0.0006 per 1K)
    Cached Input: $0.075 / 1M tokens '''

from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

client = OpenAI(api_key=openai_key)

MAX_RETRIES = 4
RETRY_BACKOFF_S = 2.0


def prompt(fen_before: str, move: str, fen_after: str) -> str:
    prompt1 = f"""Given the below chess position in FEN string format, the move played in UCI format, and the resulting FEN string after the move is played, generate expert chess commentary for this position and move like as if you are commentating a live game. Please make it as human as possible, and do your best to avoid hallucinating things going on in the position while keeping track of attacks, threats, hanging pieces, ideas, etc. Also priority keeping very good clarity and especially conciseness, 1-3 sentences with a 150 character cap including spaces and punctuation.
Before: {fen_before}
Move: {move}
After: {fen_after}
"""

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert chess commentator."},
                    {"role": "user", "content": prompt1},
                ],
            )
            text = response.choices[0].message.content
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("OpenAI returned empty response.")
            return text.strip()

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
            else:
                return f"(OpenAI error: {type(last_err).__name__})"