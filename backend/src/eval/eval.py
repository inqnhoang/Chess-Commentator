import gemini
import openAI

"""Calls to gemini/ openAI is _____.prompt """

# TO DO
def extract_eval_points (file_name) -> list[dict]:
    """ input: data.jsonl
        output: dictionary of data points for eval

        e.g. [{
            "fen_before": str,
            "move": str,
            "fen_after": str,
        }]
    """

    return

def prompt_outsource_commentary (fen_before: str, move: str, fen_after: str) -> str:
    """ prompt openAI for commentary """
    return openAI.prompt(fen_before, move, fen_after)

# TO DO
def prompt_native_commentary (fen_before: str, move: str, fen_after: str) -> str:
    """ prompt native model for commentary """
    return ""

def prompt_model_eval (fen_before: str, move: str, fen_after: str, native_output: str, outsource_output: str) -> str:
    """ prompt gemini for eval """
    return gemini.prompt(fen_before, move, fen_after, native_output, outsource_output)


def extract ():
    """ extract evaluations for eval """

    data_points = extract_eval_points("example.jsonl")

    res = []
    for data_point in data_points:
        outsource_output = prompt_outsource_commentary(
            data_point["fen_before"],
            data_point["move"],
            data_point["fen_after"]
        )
        # TO DO
        native_output = prompt_native_commentary()

        output_eval = prompt_model_eval(
            data_point["fen_before"],
            data_point["move"],
            data_point["fen_after"],
            native_output,
            outsource_output
        )

        out = data_point.copy()
        out.update({"eval": output_eval})
        res.append(out)
    
    return res

def eval ():
    evals = extract()

    # TO DO 
    # need to transfer to csv file or some convenient storage
    # perform statistical analysis e.g. percentages of prefer A, B or TIE
    # generate histograms of evals because there are several categories rated 1 - 5 for each native model and out sourced model
    # features are in order : FAITHFULNESS TO PROVIDED CONTEXT (ACCURACY / NON-HALLUCINATION), 
    #                         RELEVANCE AND FOCUS, 
    #                         INFORMATIVENESS AND SPECIFICITY (ANALYTICAL CONTENT), 
    #                         CLARITY AND FLUENCY (READABILITY), 
    #                         HUMAN-LIKENESS (NATURAL COMMENTARY STYLE), 
    #                         CONCISENESS (NO RAMBLING)

    for eval in evals:
        continue