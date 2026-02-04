from enums.board import Color, PieceType

class Piece:
    """Represents a chess piece on the board."""
    def __init__(self, piece_type: PieceType, color: Color):
        self.type = piece_type
        self.color = color

    def __repr__(self):
        return f"{self.color.name[0]}{self.type.name[0]}"  # e.g., WP, BK


class GameState:
    """Stores the current board state parsed from a FEN string."""
    def __init__(self, fen: str):
        self.fen = fen
        self.side_to_move: Color | None = None
        self.grid: list[list[Piece | None]] = [[None for _ in range(8)] for _ in range(8)]
        self._parse_fen()

    def _parse_fen(self):
        """Parses the FEN string into side_to_move and board grid."""
        parts = self.fen.split()
        if len(parts) < 2:
            raise ValueError("Invalid FEN string")

        # Side to move
        self.side_to_move = Color.WHITE if parts[1] == "w" else Color.BLACK

        # Board layout
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError("FEN must have 8 rows")

        for r, row in enumerate(rows):
            c = 0
            for char in row:
                if char.isdigit():
                    c += int(char)  # empty squares
                else:
                    color = Color.WHITE if char.isupper() else Color.BLACK
                    piece_type = self._piece_from_char(char.lower())
                    self.grid[r][c] = Piece(piece_type, color)
                    c += 1


    @staticmethod
    def _piece_from_char(c: str) -> PieceType:
        """Maps FEN character to PieceType enum."""
        mapping = {
            "p": PieceType.PAWN,
            "n": PieceType.KNIGHT,
            "b": PieceType.BISHOP,
            "r": PieceType.ROOK,
            "q": PieceType.QUEEN,
            "k": PieceType.KING
        }
        return mapping[c]


    def print_board(self):
        """Prints a simple text representation of the board."""
        for row in self.grid:
            print(" ".join([str(piece) if piece else "--" for piece in row]))
