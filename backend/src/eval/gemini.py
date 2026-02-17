''' LLM as a Judge
    Input Tokens: 
    $0.075 per 1 million tokens.
    Output Tokens: 
    $0.30 per 1 million tokens. '''

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=google_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def prompt (fen_before: str, fen_after: str, move: str, native_output: str, ai_output: str) -> str:
    prompt1 = f"""Given this chess position:
        Before: {fen_before}
        Move: {move}
        After: {fen_after}

        Commentary A: {native_output}

        Commentary B: {ai_output}

        Which commentary better describes this position in terms of accuracy, naturalness, relevance, and engagement?
        Respond with only: A or B"""
    
    response1 = model.generate_content(prompt1)
    verdict1 = response1.text.strip().upper()
    # SWAP

    prompt2 = f"""Given this chess position:
        Before: {fen_before}
        Move: {move}
        After: {fen_after}

        Commentary A: {ai_output}

        Commentary B: {native_output}

        Which commentary better describes this position in terms of accuracy, naturalness, relevance, and engagement?
        Respond with only: A or B"""
    
    response2 = model.generate_content(prompt2)
    verdict2 = response2.text.strip().upper()


    if verdict1 == "A" and verdict2 == "B":  # Both say native wins
        return "native"
    elif verdict1 == "B" and verdict2 == "A":  # Both say ai wins
        return "ai"
    else:
        return "tie"
