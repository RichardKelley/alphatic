import unittest
from alphatic.game import State, lines
from alphatic.search import Search, TacticalEvaluator


def position(x=(), o=(), player=1, k=4):
    board = [0] * 36
    for move in x: board[move] = 1
    for move in o: board[move] = -1
    return State(tuple(board), player, k)


class RulesTests(unittest.TestCase):
    def test_all_winning_windows(self):
        for k in range(3, 7):
            for line in lines(k):
                for player in (-1, 1):
                    board = [0] * 36
                    for move in line: board[move] = player
                    state = State(tuple(board), -player, k)
                    self.assertEqual(state.result(), player)
                    with self.assertRaises(ValueError): state.play(next(i for i in range(36) if not board[i]))

    def test_moves_are_immutable_and_validated(self):
        old = State()
        new = old.play(7)
        self.assertEqual(old.board[7], 0)
        self.assertEqual(new.board[7], 1)
        self.assertEqual(new.player, -1)
        for move in (-1, 36, 7):
            with self.assertRaises(ValueError): new.play(move)

    def test_no_row_wrapping(self):
        self.assertIsNone(position(x=(4, 5, 6, 7)).result())

    def test_draw(self):
        board = tuple(1 if (r + c//2) % 2 else -1 for r in range(6) for c in range(6))
        self.assertEqual(State(board, k=6).result(), 0)


class SearchTests(unittest.TestCase):
    def test_wins_and_blocks_for_either_side(self):
        for player in (-1, 1):
            for attack in (True, False):
                board = [0] * 36
                for move in (12, 13, 14): board[move] = player if attack else -player
                state = State(tuple(board), player)
                search = Search(state, TacticalEvaluator())
                search.run(80)
                self.assertEqual(search.best_move(), 15)
                self.assertEqual(search.root.visits, 80)
                if attack:
                    # The winning child's next player has lost.
                    self.assertEqual(search.root.children[15].value, -1)

    def test_priors_and_all_legal_branches(self):
        state = State().play(14)
        priors, value = TacticalEvaluator().evaluate(state)
        self.assertAlmostEqual(sum(priors.values()), 1)
        self.assertEqual(set(priors), set(state.legal_moves()))
        self.assertTrue(-1 <= value <= 1)
        search = Search(state)
        search.run(15)
        self.assertEqual(sum(n.visits for n in search.root.children.values()), 14)
        self.assertIn(search.best_move(), state.legal_moves())

    def test_terminal_search(self):
        search = Search(position(x=(0, 1, 2, 3), player=-1))
        search.step()
        self.assertEqual(search.root.value, -1)
        self.assertFalse(search.root.children)
        with self.assertRaises(ValueError): search.best_move()

    def test_complete_self_play(self):
        state = State()
        for _ in range(36):
            if state.result() is not None: break
            search = Search(state)
            search.run(20)
            state = state.play(search.best_move())
        self.assertIsNotNone(state.result())


if __name__ == '__main__': unittest.main()
