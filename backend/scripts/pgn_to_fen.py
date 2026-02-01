from pathlib import Path
import json
import chess.pgn

NUM_GAMES = 3  # first N games from the PGN
INCLUDE_START_POS = True  # include initial board position as ply=0
OUTPUT_FORMAT = "fen"  # whether "jsonl", "json", or "fen" is desired

IN_PGN_REL = Path("data/raw/games.pgn")

# Output directory
OUT_DIR_REL = Path("data/fen_sequences")


def repo_root():
    return Path(__file__).resolve().parents[2]


def safe_game_id(game: chess.pgn.Game, idx: int) -> str:
    event = (game.headers.get("Event") or "game").strip().replace(" ", "_")
    date = (game.headers.get("Date") or "????.??.??").strip().replace(".", "-")
    white = (game.headers.get("White") or "White").strip().replace(" ", "_")
    black = (game.headers.get("Black") or "Black").strip().replace(" ", "_")
    return f"{idx:04d}_{date}_{event}_{white}_vs_{black}"[:120]


def extract_sequence(game: chess.pgn.Game, include_start: bool) -> list[dict]:
    """
    Returns a list of dicts in ply order:
      {"ply": int, "san": str | None, "fen": str}
    """
    board = game.board()
    seq: list[dict] = []

    if include_start:
        seq.append({"ply": 0, "san": None, "fen": board.fen()})

    ply = 0
    for move in game.mainline_moves():
        ply += 1
        san = board.san(move)
        board.push(move)
        seq.append({"ply": ply, "san": san, "fen": board.fen()})

    return seq


def write_jsonl(path: Path, seq: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in seq:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, seq: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(seq, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_fen_lines(path: Path, seq: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in seq:
            f.write(row["fen"] + "\n")


def main():
    root = repo_root()
    in_pgn = root / IN_PGN_REL
    out_dir = root / OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_pgn.exists():
        raise FileNotFoundError(f"PGN not found at: {in_pgn}")

    # deletes all existing sequence files
    for f in out_dir.glob("*"):
        if f.is_file():
            f.unlink()

    games_read = 0
    total_positions = 0

    with in_pgn.open("r") as pgn:
        while games_read < NUM_GAMES:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            gid = safe_game_id(game, games_read + 1)
            seq = extract_sequence(game, INCLUDE_START_POS)

            fmt = OUTPUT_FORMAT.lower()

            if fmt == "jsonl":
                out_path = out_dir / f"{gid}.jsonl"
                write_jsonl(out_path, seq)
            elif fmt == "json":
                out_path = out_dir / f"{gid}.json"
                write_json(out_path, seq)
            elif fmt == "fen":
                out_path = out_dir / f"{gid}.fen.txt"
                write_fen_lines(out_path, seq)
            else:
                raise ValueError('OUTPUT_FORMAT must be "jsonl", "json", or "fen"')

            games_read += 1
            total_positions += len(seq)

    print(f"Games processed: {games_read}")
    print(f"Total positions written: {total_positions}")
    print(f"Format: {OUTPUT_FORMAT}")


if __name__ == "__main__":
    main()
