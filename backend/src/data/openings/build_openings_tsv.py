from __future__ import annotations

import os
from pathlib import Path
from datasets import load_dataset

def normalize_uci(uci_field) -> str:
    """
    Normalize dataset 'uci' field to space-separated string.
    """
    if uci_field is None:
        return ""
    if isinstance(uci_field, str):
        return uci_field.strip()
    if isinstance(uci_field, (list, tuple)):
        return " ".join(str(x) for x in uci_field).strip()
    return str(uci_field).strip()

def main():
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "openings.tsv"

    print(f"Building openings.tsv in: {out_path}")

    # load hugging face dataset
    print("Loading Lichess/chess-openings dataset...")
    ds = load_dataset("Lichess/chess-openings", split="train")

    print("Dataset columns:", ds.column_names)

    required = ["eco", "name", "uci"]
    for col in required:
        if col not in ds.column_names:
            raise RuntimeError(
                f"Expected column '{col}' not found.\n"
                f"Available columns: {ds.column_names}"
            )

    # write tsv file
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            eco = (row.get("eco") or "").strip()
            name = (row.get("name") or "").strip()
            uci = normalize_uci(row.get("uci"))

            if not eco or not name or not uci:
                continue

            uci = " ".join(uci.split())
            f.write(f"{eco}\t{name}\t{uci}\n")
            count += 1

    print(f"Done. Wrote {count} openings.")
    print(f"File location: {out_path}")

if __name__ == "__main__":
    main()
