import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import argparse
import unittest
import pygame
from alphatic.app import App
from alphatic.network import Network


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = App(argparse.Namespace(win_length=4, simulations=50, autoplay=False, model=None))
        self.app.draw()

    def tearDown(self):
        pygame.quit()

    def test_play_inspect_pause_step_undo(self):
        app = self.app
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(104, 266), button=1))
        self.assertTrue(app.thinking)
        self.assertEqual(app.state.board[0], 1)
        app.action('pause')
        app.update()
        self.assertEqual(app.search.root.visits, 0)
        app.action('step')
        self.assertEqual(app.search.root.visits, 1)
        app.search.run(49)
        app.draw()
        child_rect, child = app.node_hits[1]
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=child_rect.center, button=1))
        self.assertIs(app.focus, child)
        app.draw()
        app.action('up')
        self.assertIs(app.focus, app.search.root)
        app.action('play')
        self.assertFalse(app.thinking)
        self.assertEqual(sum(v != 0 for v in app.state.board), 2)
        app.draw()
        app.action('undo')
        self.assertEqual(app.state.board, (0,)*36)
        self.assertIsNone(app.search)

    def test_ai_first_autoplay_reset_and_render(self):
        app = self.app
        self.assertIsInstance(app.evaluator, Network)
        self.assertGreater(app.evaluator.games, 0)
        app.action('first')
        self.assertEqual(app.human, -1)
        self.assertTrue(app.thinking)
        app.search.run(60)
        app.draw()
        app.action('play')
        self.assertFalse(app.thinking)
        app.action('auto')
        self.assertTrue(app.thinking)
        app.action('reset')
        self.assertEqual(app.state.board, (0,)*36)
        app.draw()
