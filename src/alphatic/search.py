"""PUCT search; values are stored from each node's player-to-move view."""
from dataclasses import dataclass, field
from math import sqrt, tanh
from typing import Protocol
from .game import State, lines


class Evaluator(Protocol):
    def evaluate(self, state: State) -> tuple[dict[int, float], float]: ...


class TacticalEvaluator:
    """Untrained policy/value baseline based on open winning windows."""
    name = "Tactical policy + value"

    def evaluate(self, state: State) -> tuple[dict[int, float], float]:
        legal = state.legal_moves()
        scores = {m: 1.0 + (5 - abs(m // 6 - 2.5) - abs(m % 6 - 2.5)) * .12 for m in legal}
        wins, blocks = set(), set()
        balance = 0.0
        for window in lines(state.k):
            own = sum(state.board[i] == state.player for i in window)
            opp = sum(state.board[i] == -state.player for i in window)
            if own and opp:
                continue
            empty = [i for i in window if not state.board[i]]
            if own:
                balance += 3 ** own
            if opp:
                balance -= 3 ** opp
            for m in empty:
                scores[m] += (4 ** own if own else 0) + .85 * (4 ** opp if opp else 0)
            if len(empty) == 1:
                if own == state.k - 1:
                    wins.add(empty[0])
                if opp == state.k - 1:
                    blocks.add(empty[0])
        # Keep every legal branch, but strongly prioritize forced tactics.
        forced = wins or blocks
        if forced:
            total = sum(scores.values())
            scores = {m: .002 * v / total + (.998 / len(forced) if m in forced else 0) for m, v in scores.items()}
        else:
            total = sum(scores.values())
            scores = {m: v / total for m, v in scores.items()}
        value = .98 if wins else (-.98 if len(blocks) > 1 else tanh(balance / 100))
        return scores, value


@dataclass(slots=True, eq=False)
class Node:
    state: State
    prior: float = 1.0
    move: int | None = None
    parent: "Node | None" = field(default=None, repr=False)
    children: dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def ranked(self) -> list["Node"]:
        return sorted(self.children.values(), key=lambda n: (n.visits, n.prior), reverse=True)


class Search:
    def __init__(self, state: State, evaluator: Evaluator | None = None, exploration: float = 1.6):
        self.root = Node(state)
        if evaluator is None:
            from .network import default_network
            evaluator = default_network(state.k)
        self.evaluator = evaluator
        self.exploration = exploration
        self.last_path: list[Node] = []
        self.nodes = 1
        self.max_depth = 0

    def step(self) -> None:
        node = self.root
        path = [node]
        while node.children:
            scale = self.exploration * sqrt(max(1, node.visits))
            node = max(node.children.values(), key=lambda child: -child.value + scale * child.prior / (1 + child.visits))
            path.append(node)
        result = node.state.result()
        if result is not None:
            value = float(result * node.state.player)
        else:
            priors, value = self.evaluator.evaluate(node.state)
            for move, prior in priors.items():
                node.children[move] = Node(node.state.play(move), prior, move, node)
            self.nodes += len(node.children)
        for item in reversed(path):
            item.visits += 1
            item.value_sum += value
            value = -value
        self.last_path = path
        self.max_depth = max(self.max_depth, len(path) - 1)

    def run(self, simulations: int) -> None:
        for _ in range(simulations):
            self.step()

    def best_move(self) -> int:
        if self.root.state.result() is not None:
            raise ValueError("cannot choose a move in a finished game")
        if not self.root.children:
            self.step()
        return self.root.ranked()[0].move
