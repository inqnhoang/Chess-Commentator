''' Generate AI commentary
    Input Cost: $0.15 / 1M tokens ($0.00015 per 1K)
    Output Cost: $0.60 / 1M tokens ($0.0006 per 1K)
    Cached Input: $0.075 / 1M tokens '''

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

def prompt (fen_before: str, move: str, fen_after: str) -> str:
    prompt1 = f"""Given this chess position:
        Before: {fen_before}
        Move: {move}
        After: {fen_after}

        Generate expert chess commentary for this position and move and no extra sentences or words. 
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {   "role": "system", 
                "content": "You are an expert chess commentator."},

            {   "role": "user",
                "content": prompt1
            }
        ]
    )

    return response.choices[0].message.content
