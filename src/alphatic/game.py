"""Immutable 6 x 6 connect-k rules. X = +1, O = -1."""
from dataclasses import dataclass
from functools import lru_cache

SIZE = 6


@lru_cache(maxsize=4)
def lines(k: int) -> tuple[tuple[int, ...], ...]:
    if not 3 <= k <= SIZE:
        raise ValueError("win length must be between 3 and 6")
    return tuple(
        tuple((r + i * dr) * SIZE + c + i * dc for i in range(k))
        for r in range(SIZE)
        for c in range(SIZE)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
        if 0 <= r + (k - 1) * dr < SIZE and 0 <= c + (k - 1) * dc < SIZE
    )


@dataclass(frozen=True, slots=True)
class State:
    board: tuple[int, ...] = (0,) * 36
    player: int = 1
    k: int = 4

    def __post_init__(self):
        if len(self.board) != 36 or any(v not in (-1, 0, 1) for v in self.board):
            raise ValueError("board must have 36 cells containing -1, 0, or 1")
        if self.player not in (-1, 1):
            raise ValueError("player must be -1 or 1")
        lines(self.k)

    def winning_line(self) -> tuple[int, ...]:
        for line in lines(self.k):
            first = self.board[line[0]]
            if first and all(self.board[i] == first for i in line[1:]):
                return line
        return ()

    def result(self) -> int | None:
        line = self.winning_line()
        return self.board[line[0]] if line else (None if 0 in self.board else 0)

    def legal_moves(self) -> list[int]:
        return [i for i, cell in enumerate(self.board) if not cell]

    def play(self, move: int) -> "State":
        if not 0 <= move < 36 or self.board[move] or self.result() is not None:
            raise ValueError("illegal move")
        board = list(self.board)
        board[move] = self.player
        return State(tuple(board), -self.player, self.k)


def coordinate(move: int) -> str:
    return f"{'ABCDEF'[move % 6]}{move // 6 + 1}"
