import json
from pathlib import Path
from typing import List, Dict
from linearize import linearize_for_t5

BASE_DIR = Path(__file__).resolve().parent
DATA_JSON = BASE_DIR / "data.json"
TRAINING_JSON = BASE_DIR / "training_data.jsonl"

def create_t5_dataset(input_file: str, output_file: str):
    """
    Convert chess move JSON data to T5 training format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
    """
    # Load JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Process each example
    processed = []
    
    for i, example in enumerate(data):
        try:
            x, y = linearize_for_t5(example)
            processed.append({
                'input': x,
                'target': y
            })
        except Exception as e:
            print(f"Error on example {i}: {e}")
    
    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')
    
    # Show example
    if processed:
        print("\n=== EXAMPLE ===")
        print(f"Input:  {processed[0]['input'][:150]}...")
        print(f"Target: {processed[0]['target'][:150]}...")



if __name__ == "__main__":
    create_t5_dataset(
        input_file=str(DATA_JSON),
        output_file=str(TRAINING_JSON)
    )


