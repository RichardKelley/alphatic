import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from argparse import Namespace
import numpy as np
from alphatic.game import State
from alphatic.network import Network, encode, augment, default_network, train
from alphatic.search import Search


class NetworkTests(unittest.TestCase):
    def test_evaluation_is_entirely_neural_even_for_forced_tactics(self):
        model = Network()
        for name in ("w1", "b1", "wp", "bp", "wv", "bv"):
            getattr(model, name).fill(0)
        model.bv[0] = .3
        # X can win at D1. A uniform network must remain uniform, without overrides.
        board = [0] * 36
        board[:3] = [1, 1, 1]
        state = State(tuple(board))
        with patch('alphatic.search.TacticalEvaluator.evaluate', side_effect=AssertionError("heuristic called")):
            priors, value = model.evaluate(state)
            self.assertEqual(set(priors), set(state.legal_moves()))
            np.testing.assert_allclose(list(priors.values()), 1 / 33)
            self.assertAlmostEqual(value, float(np.tanh(.3)), places=6)
            search = Search(state, model)
            search.step()
            self.assertAlmostEqual(search.root.value, value)
            # Terminal outcomes still bypass neural estimates.
            finished = Search(state.play(3), model)
            finished.step()
            self.assertEqual(finished.root.value, -1)

    def test_default_search_uses_packaged_neural_model(self):
        model = default_network()
        self.assertIsInstance(model, Network)
        self.assertGreater(model.games, 0)
        self.assertIsInstance(Search(State()).evaluator, Network)
        self.assertEqual(default_network(6).k, 6)

    def test_self_play_uses_network_from_first_game(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trained.npz'
            args = Namespace(model=path, seed=5, win_length=4, games=1, simulations=2)
            with patch('alphatic.search.TacticalEvaluator.evaluate', side_effect=AssertionError("heuristic called")):
                train(args)
            self.assertEqual(Network.load(path).games, 1)

    def test_mask_roundtrip_and_learning(self):
        model = Network()
        state = State().play(0).play(12)
        x = encode(state)[None, :]
        _, p, _ = model.forward(x)
        self.assertEqual(p[0, 0], 0)
        self.assertEqual(p[0, 12], 0)
        target = np.zeros((1, 36), dtype=np.float32)
        target[0, 15] = 1
        losses = [model.fit(x, target, np.array([[1]], dtype=np.float32)) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.npz'
            model.save(path)
            loaded = Network.load(path)
            np.testing.assert_allclose(model.forward(x)[1], loaded.forward(x)[1])
        with self.assertRaises(ValueError): model.evaluate(State(k=5))

    def test_augmentation_preserves_legal_policy(self):
        state = State().play(0).play(7)
        policy = np.zeros(36, dtype=np.float32)
        policy[12] = 1
        samples = list(augment(state, policy))
        self.assertEqual(len(samples), 8)
        for x, p in samples:
            self.assertEqual(p.sum(), 1)
            self.assertEqual(float((x[72:] * p).sum()), 1)
