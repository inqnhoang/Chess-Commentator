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

def prompt (fen_before: str, move: str, fen_after: str, native_output: str, ai_output: str) -> list[str]:
    prompt1 = f"""Given this chess position:
        Before: {fen_before}
        Move: {move}
        After: {fen_after}

        Commentary A: {native_output}

        Commentary B: {ai_output}

        CRITERIA (6 TOTAL) — Score each 1–5 for A and for B

        1) FAITHFULNESS TO PROVIDED CONTEXT (ACCURACY / NON-HALLUCINATION)
        What it means: Commentary must be consistent with the feature context. Do not invent events (check/mate/capture/trade/hanging piece/promotion/eval swing) unless supported by the features/flags. If the context is NA/ambiguous, avoid strong claims.

        Anchors:
        5: No unsupported claims; all statements supported by feature context.
        4: Mostly faithful; one minor overreach or vague inference, not misleading.
        3: Generally aligned but at least one questionable claim/assumption.
        2: Multiple unsupported claims OR one major factual error.
        1: Blatantly contradicts context or heavily hallucinates key events.

        DISQUALIFYING ERRORS (HARD RULE)
        If a candidate claims any of the following WITHOUT feature support, set Faithfulness = 1:
        - check / checkmate / capture / promotion / trade / hanging-piece win
        - a forced tactic or specific threat that is not indicated by context
        If both candidates have disqualifying errors, both can receive Faithfulness=1; winner is decided by totals, but neither is “good.”


        2) RELEVANCE AND FOCUS
        What it means: Commentary should discuss the current move and the most salient consequences from context (eval swing, blunder/mistake signal, hanging value, trades, checks, king pressure).

        Anchors:
        5: Directly addresses move + key salient signals.
        4: Mostly focused; small generic portion but clearly tied to context.
        3: Some relevance but misses the most important signal or too generic.
        2: Mostly generic; weak connection to context.
        1: Unrelated to move/context.

        3) INFORMATIVENESS AND SPECIFICITY (ANALYTICAL CONTENT)
        What it means: Provides concrete, useful information grounded in context (material change, tactical danger, king safety, hanging value, trade type, evaluation direction) without inventing details.

        Anchors:
        5: Specific and informative; 1–2 clear grounded takeaways.
        4: Informative but slightly less specific or slightly repetitive.
        3: Some useful info but vague (“good move”) or incomplete.
        2: Low info; mostly fluff or unclear why the move matters.
        1: No meaningful content beyond generic praise/criticism.

        4) CLARITY AND FLUENCY (READABILITY)
        What it means: Easy to read; grammatical and coherent.

        Anchors:
        5: Very clear, fluent, well-structured.
        4: Readable with minor awkward phrasing.
        3: Understandable but noticeably awkward/confusing at times.
        2: Hard to read; multiple grammar/structure issues.
        1: Incoherent/unreadable.

        5) HUMAN-LIKENESS (NATURAL COMMENTARY STYLE)
        What it means: Sounds like a human commentator rather than a rigid template. This is about phrasing and flow, not adding extra facts.

        Anchors:
        5: Natural, varied, commentator-like, not templated.
        4: Mostly natural; slight template feel.
        3: Noticeably templated/robotic but acceptable.
        2: Very robotic or repetitive.
        1: Unnatural to the point of unusable.

        6) CONCISENESS (NO RAMBLING)
        What it means: Compact and not overlong. Prefer 1–2 sentences unless context clearly warrants more. Avoid repeating the same point.

        Anchors:
        5: Concise and complete; no unnecessary text.
        4: Slightly long or slightly abrupt, but efficient.
        3: Some extra fluff or mild repetition.
        2: Rambling OR too short to convey meaning.
        1: Severely rambling OR essentially empty.

        AGGREGATION AND DECISION RULE
        1) Score A and B on each criterion (1–5).
        2) Compute weighted totals to prioritize accuracy:
        Total(A) = Faithfulness + Relevance + Informativeness + Clarity + Human-likeness + Conciseness
        Total(B) = Faithfulness + Relevance + Informativeness + Clarity + Human-likeness + Conciseness
        3) Winner = higher total.

        TIE-BREAK RULES
        If totals tie:
        1) Prefer higher Faithfulness to Provided Context
        2) If still tied, prefer higher Relevance and Focus
        3) If still tied, prefer higher Informativeness and Specificity
        4) If still tied, declare Tie

        NOTES TO REDUCE BIAS IN JUDGING
        - Score A and B independently first before choosing a winner (reduces anchoring).
        - Do not reward confident tone if faithfulness is weak.
        - If context lacks evidence for a claim, prefer cautious phrasing and score faithfulness higher.
        - Don’t reward extra detail if it isn’t supported by context.

        OUTPUT FORMAT (CSV - ONE LINE)
        Return one line in this format:
        WINNER,<A|B|TIE>,A,<C1>,<C2>,<C3>,<C4>,<C5>,<C6>,<A_TOTAL>,B,<C1>,<C2>,<C3>,<C4>,<C5>,<C6>,<B_TOTAL>
        """
    
    response1 = model.generate_content(prompt1)
    verdict1 = response1.text.strip().upper()
    # SWAP

    prompt2 = f"""Given this chess position:
        Before: {fen_before}
        Move: {move}
        After: {fen_after}

        Commentary A: {ai_output}

        Commentary B: {native_output}

        CRITERIA (6 TOTAL) — Score each 1–5 for A and for B

        1) FAITHFULNESS TO PROVIDED CONTEXT (ACCURACY / NON-HALLUCINATION)
        What it means: Commentary must be consistent with the feature context. Do not invent events (check/mate/capture/trade/hanging piece/promotion/eval swing) unless supported by the features/flags. If the context is NA/ambiguous, avoid strong claims.

        Anchors:
        5: No unsupported claims; all statements supported by feature context.
        4: Mostly faithful; one minor overreach or vague inference, not misleading.
        3: Generally aligned but at least one questionable claim/assumption.
        2: Multiple unsupported claims OR one major factual error.
        1: Blatantly contradicts context or heavily hallucinates key events.

        DISQUALIFYING ERRORS (HARD RULE)
        If a candidate claims any of the following WITHOUT feature support, set Faithfulness = 1:
        - check / checkmate / capture / promotion / trade / hanging-piece win
        - a forced tactic or specific threat that is not indicated by context
        If both candidates have disqualifying errors, both can receive Faithfulness=1; winner is decided by totals, but neither is “good.”


        2) RELEVANCE AND FOCUS
        What it means: Commentary should discuss the current move and the most salient consequences from context (eval swing, blunder/mistake signal, hanging value, trades, checks, king pressure).

        Anchors:
        5: Directly addresses move + key salient signals.
        4: Mostly focused; small generic portion but clearly tied to context.
        3: Some relevance but misses the most important signal or too generic.
        2: Mostly generic; weak connection to context.
        1: Unrelated to move/context.

        3) INFORMATIVENESS AND SPECIFICITY (ANALYTICAL CONTENT)
        What it means: Provides concrete, useful information grounded in context (material change, tactical danger, king safety, hanging value, trade type, evaluation direction) without inventing details.

        Anchors:
        5: Specific and informative; 1–2 clear grounded takeaways.
        4: Informative but slightly less specific or slightly repetitive.
        3: Some useful info but vague (“good move”) or incomplete.
        2: Low info; mostly fluff or unclear why the move matters.
        1: No meaningful content beyond generic praise/criticism.

        4) CLARITY AND FLUENCY (READABILITY)
        What it means: Easy to read; grammatical and coherent.

        Anchors:
        5: Very clear, fluent, well-structured.
        4: Readable with minor awkward phrasing.
        3: Understandable but noticeably awkward/confusing at times.
        2: Hard to read; multiple grammar/structure issues.
        1: Incoherent/unreadable.

        5) HUMAN-LIKENESS (NATURAL COMMENTARY STYLE)
        What it means: Sounds like a human commentator rather than a rigid template. This is about phrasing and flow, not adding extra facts.

        Anchors:
        5: Natural, varied, commentator-like, not templated.
        4: Mostly natural; slight template feel.
        3: Noticeably templated/robotic but acceptable.
        2: Very robotic or repetitive.
        1: Unnatural to the point of unusable.

        6) CONCISENESS (NO RAMBLING)
        What it means: Compact and not overlong. Prefer 1–2 sentences unless context clearly warrants more. Avoid repeating the same point.

        Anchors:
        5: Concise and complete; no unnecessary text.
        4: Slightly long or slightly abrupt, but efficient.
        3: Some extra fluff or mild repetition.
        2: Rambling OR too short to convey meaning.
        1: Severely rambling OR essentially empty.

        AGGREGATION AND DECISION RULE
        1) Score A and B on each criterion (1–5).
        2) Compute weighted totals to prioritize accuracy:
        Total(A) = Faithfulness + Relevance + Informativeness + Clarity + Human-likeness + Conciseness
        Total(B) = Faithfulness + Relevance + Informativeness + Clarity + Human-likeness + Conciseness
        3) Winner = higher total.

        TIE-BREAK RULES
        If totals tie:
        1) Prefer higher Faithfulness to Provided Context
        2) If still tied, prefer higher Relevance and Focus
        3) If still tied, prefer higher Informativeness and Specificity
        4) If still tied, declare Tie

        NOTES TO REDUCE BIAS IN JUDGING
        - Score A and B independently first before choosing a winner (reduces anchoring).
        - Do not reward confident tone if faithfulness is weak.
        - If context lacks evidence for a claim, prefer cautious phrasing and score faithfulness higher.
        - Don’t reward extra detail if it isn’t supported by context.

        OUTPUT FORMAT (CSV - ONE LINE)
        Return one line in this format:
        WINNER,<A|B|TIE>,A,<C1>,<C2>,<C3>,<C4>,<C5>,<C6>,<A_TOTAL>,B,<C1>,<C2>,<C3>,<C4>,<C5>,<C6>,<B_TOTAL>
        """
    
    response2 = model.generate_content(prompt2)
    verdict2 = response2.text.strip().upper()

    return [verdict1, verdict2]
