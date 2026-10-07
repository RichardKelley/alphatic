"""Pygame board and live, navigable search-tree inspector."""
import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from .game import State, coordinate
from .search import Search
from .network import Network, default_network

BG = (13, 19, 29)
PANEL = (21, 30, 44)
CARD = (29, 41, 57)
LINE = (48, 65, 83)
TEXT = (231, 238, 246)
MUTED = (143, 162, 184)
TEAL = (80, 221, 184)
CORAL = (255, 151, 126)
GOLD = (246, 208, 120)
W, H = 1440, 900


class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("AlphaTic · 6 × 6 search lab")
        self.window = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.canvas = pygame.Surface((W, H))
        self.fonts = {size: pygame.font.SysFont("arial", size) for size in (12, 14, 16, 18, 20, 24, 30, 40)}
        self.args = args
        self.evaluator = Network.load(args.model) if args.model else default_network(args.win_length)
        if self.evaluator.k != args.win_length:
            raise ValueError("model win length does not match --win-length")
        self.budget = args.simulations
        self.autoplay = args.autoplay
        self.paused = False
        self.heatmap = True
        self.human = 1
        self.buttons = []
        self.node_hits = []
        self.page = 0
        self.reset()

    def reset(self):
        self.state = State(k=self.args.win_length)
        self.history = []
        self.search = None
        self.focus = None
        self.thinking = False
        self.last_move = None
        self.page = 0
        self.maybe_start()

    def maybe_start(self):
        if self.state.result() is None and (self.autoplay or self.state.player != self.human):
            self.search = Search(self.state, self.evaluator)
            self.focus = self.search.root
            self.page = 0
            self.thinking = True
            self.started = time.monotonic()

    def play(self, move):
        self.history.append((self.state, self.last_move))
        self.state = self.state.play(move)
        self.last_move = move
        self.thinking = False
        self.maybe_start()

    def commit(self):
        if self.thinking and self.search:
            self.play(self.search.best_move())

    def undo(self):
        if not self.history:
            return
        self.state, self.last_move = self.history.pop()
        if self.state.player != self.human and self.history:
            self.state, self.last_move = self.history.pop()
        self.search = self.focus = None
        self.thinking = False
        self.autoplay = False
        self.maybe_start()

    def update(self):
        if not self.thinking or self.paused:
            return
        deadline = time.monotonic() + .010
        while self.search.root.visits < self.budget and time.monotonic() < deadline:
            self.search.step()
        if self.search.root.visits >= self.budget and time.monotonic() - self.started >= 1.2:
            self.commit()

    def text(self, message, pos, size=18, color=TEXT, center=False):
        rendered = self.fonts[size].render(str(message), True, color)
        rect = rendered.get_rect(center=pos) if center else rendered.get_rect(topleft=pos)
        self.canvas.blit(rendered, rect)

    def panel(self, rect, color=PANEL, radius=16, border=None):
        pygame.draw.rect(self.canvas, color, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.canvas, border, rect, 1, border_radius=radius)

    def button(self, label, rect, action, active=False):
        self.panel(rect, (35, 75, 72) if active else CARD, 8, TEAL if active else LINE)
        self.text(label, pygame.Rect(rect).center, 16, TEAL if active else TEXT, True)
        self.buttons.append((pygame.Rect(rect), action))

    def mark(self, player, center, radius, width=4):
        x, y = center
        if player == 1:
            pygame.draw.line(self.canvas, TEAL, (x-radius, y-radius), (x+radius, y+radius), width)
            pygame.draw.line(self.canvas, TEAL, (x+radius, y-radius), (x-radius, y+radius), width)
        elif player == -1:
            pygame.draw.circle(self.canvas, CORAL, center, radius, width)

    def mini_board(self, state, rect, move=None):
        x, y, size, _ = rect
        cell = size / 6
        for i in range(36):
            cx, cy = x + (i % 6) * cell, y + (i // 6) * cell
            if i == move:
                pygame.draw.rect(self.canvas, (58, 76, 84), (cx, cy, cell, cell))
            if state.board[i]:
                self.mark(state.board[i], (int(cx+cell/2), int(cy+cell/2)), max(2, int(cell*.27)), 2)
        for n in range(1, 6):
            pygame.draw.line(self.canvas, LINE, (x+n*cell, y), (x+n*cell, y+size))
            pygame.draw.line(self.canvas, LINE, (x, y+n*cell), (x+size, y+n*cell))

    def tree_card(self, node, rect, root=False):
        x, y, w, h = rect
        on_path = self.search and node in self.search.last_path
        self.panel(rect, CARD, 10, GOLD if on_path else LINE)
        mover = -node.state.player
        label = "Search root" if node.parent is None else f"{'X' if mover == 1 else 'O'}  {coordinate(node.move)}"
        self.text(label, (x+w/2, y+17), 16, TEXT, True)
        self.mini_board(node.state, (x+(w-66)/2, y+35, 66, 66), node.move)
        # Consistent perspective across the entire displayed tree.
        q = node.value * node.state.player * self.search.root.state.player
        self.text(f"N {node.visits:,}", (x+w/2, y+116), 14, TEAL, True)
        self.text(f"Q {q:+.2f}" if node.visits else "Q --", (x+w/2, y+136), 14, MUTED, True)
        if not root:
            self.text(f"P {node.prior:.1%}", (x+w/2, y+156), 12, MUTED, True)
        self.node_hits.append((pygame.Rect(rect), node))

    def draw_board(self):
        self.panel((24, 108, 592, 682))
        result = self.state.result()
        symbol = 'X' if self.state.player == 1 else 'O'
        if result is not None:
            status = "Draw · all squares filled" if result == 0 else f"{'X' if result == 1 else 'O'} wins · connect {self.state.k}"
        elif self.thinking:
            status = f"{symbol} is thinking" + (" · paused" if self.paused else "…")
        else:
            status = f"Your turn · {symbol}"
        self.text(status, (48, 129), 24)
        self.text(f"6 × 6 board    /    {self.state.k} in a row wins", (48, 164), 16, MUTED)
        bx, by, cell = 62, 224, 84
        root = self.search.root if self.search else None
        show_heat = self.heatmap and root and root.state == self.state
        max_n = max((n.visits for n in root.children.values()), default=1) if show_heat else 1
        for i in range(6):
            self.text("ABCDEF"[i], (bx+i*cell+42, by-19), 14, MUTED, True)
            self.text(i+1, (bx-17, by+i*cell+42), 14, MUTED, True)
        for move, player in enumerate(self.state.board):
            x, y = bx + move % 6 * cell, by + move // 6 * cell
            rect = (x+3, y+3, cell-6, cell-6)
            child = root.children.get(move) if show_heat else None
            n = child.visits if child else 0
            heat = n / max(1, max_n)
            color = (int(27+15*heat), int(39+65*heat), int(55+40*heat))
            self.panel(rect, color, 8, GOLD if move == self.last_move else None)
            if player:
                self.mark(player, (x+42, y+42), 19, 5)
            elif child and n:
                self.text(f"{n}", (x+42, y+40), 18, TEAL, True)
                self.text("visits", (x+42, y+60), 12, MUTED, True)
        win = self.state.winning_line()
        if win:
            a, b = win[0], win[-1]
            pygame.draw.line(self.canvas, GOLD, (bx+a%6*cell+42, by+a//6*cell+42), (bx+b%6*cell+42, by+b//6*cell+42), 5)
        caption = "Brighter squares = more search visits" if show_heat else "Click an empty square to play"
        if result is not None:
            caption = "New game to play again · tree remains inspectable"
        self.text(caption, (48, 751), 16, MUTED)

    def draw_tree(self):
        self.panel((636, 108, 780, 682))
        self.text("Inside the search", (660, 129), 24)
        self.text("Click a node to explore its continuations", (660, 164), 16, MUTED)
        if not self.search:
            self.text("The tree starts growing on the AI's turn.", (1026, 393), 20, MUTED, True)
            self.text("Play a square, or choose AI first.", (1026, 427), 18, MUTED, True)
            return
        root, focus = self.search.root, self.focus
        current = "LIVE" if self.thinking else "LAST SEARCH"
        self.text(current, (1280, 135), 14, TEAL)
        self.text(f"{root.visits:,} simulations   /   {self.search.nodes:,} nodes   /   depth {self.search.max_depth}", (660, 196), 16, TEAL)
        self.button("Root", (1190, 230, 90, 32), "root")
        self.button("Up", (1290, 230, 90, 32), "up")
        depth, ancestor = 0, focus
        while ancestor.parent:
            depth += 1
            ancestor = ancestor.parent
        self.text(f"Viewing depth {depth}", (660, 237), 14, MUTED)
        self.tree_card(focus, (956, 270, 140, 152), True)
        ranked = focus.ranked()
        pages = max(1, (len(ranked)+4)//5)
        self.page = min(self.page, pages-1)
        children = ranked[self.page*5:self.page*5+5]
        for index, child in enumerate(children):
            x = 660 + index*146
            pygame.draw.line(self.canvas, LINE, (1026, 422), (x+66, 473), 2)
            self.tree_card(child, (x, 473, 132, 172))
            for j, grandchild in enumerate(child.ranked()[:2]):
                gx = x + j*70
                pygame.draw.line(self.canvas, LINE, (x+66, 645), (gx+31, 684), 2)
                self.panel((gx, 684, 62, 44), CARD, 6, GOLD if grandchild in self.search.last_path else LINE)
                self.text(coordinate(grandchild.move), (gx+31, 696), 12, TEXT, True)
                self.text(f"N {grandchild.visits}", (gx+31, 715), 12, TEAL, True)
                self.node_hits.append((pygame.Rect(gx, 684, 62, 44), grandchild))
        if not children:
            message = "Terminal position" if focus.state.result() is not None else "Not expanded yet"
            self.text(message, (1026, 517), 20, MUTED, True)
        self.button("<", (660, 743, 36, 28), "prev")
        self.button(">", (702, 743, 36, 28), "next")
        self.text(f"{self.page+1}/{pages} · {len(ranked)} branches, sorted by visits", (748, 749), 14, MUTED)
        self.text("Gold: latest path", (1242, 749), 14, GOLD)

    def draw(self):
        self.canvas.fill(BG)
        self.buttons, self.node_hits = [], []
        self.text("AlphaTic", (28, 22), 40)
        self.text(self.evaluator.name, (1010, 32), 18, TEAL)
        self.text("PUCT · policy-guided Monte Carlo tree search", (1010, 59), 14, MUTED)
        self.draw_board()
        self.draw_tree()
        self.button("New game [R]", (24, 806, 144, 40), "reset")
        self.button("AI first [F]", (178, 806, 124, 40), "first")
        self.button("Undo [U]", (312, 806, 102, 40), "undo")
        self.button("Auto [A]", (424, 806, 100, 40), "auto", self.autoplay)
        self.button("Heat [H]", (534, 806, 100, 40), "heat", self.heatmap)
        self.button("Resume [Space]" if self.paused else "Pause [Space]", (654, 806, 158, 40), "pause", self.paused)
        self.button("Step [S]", (822, 806, 100, 40), "step")
        self.button("Play [Enter]", (932, 806, 126, 40), "play")
        self.button("−", (1078, 806, 40, 40), "less")
        self.text(f"{self.budget:,} sims", (1188, 826), 16, TEXT, True)
        self.button("+", (1260, 806, 40, 40), "more")
        perspective = 'X' if self.search and self.search.root.state.player == 1 else 'O'
        self.text(f"N = visits   P = policy prior   Q = value for searching {perspective} (−1 loss, +1 win; estimate, not probability)", (28, 866), 14, MUTED)
        width, height = self.window.get_size()
        scale = min(width/W, height/H)
        size = (max(1, int(W*scale)), max(1, int(H*scale)))
        self.offset = ((width-size[0])//2, (height-size[1])//2)
        self.scale = scale
        self.window.fill(BG)
        self.window.blit(pygame.transform.smoothscale(self.canvas, size), self.offset)
        pygame.display.flip()

    def action(self, action):
        if action == "reset":
            self.human, self.autoplay, self.paused = 1, False, False
            self.reset()
        elif action == "first":
            self.human, self.autoplay, self.paused = -1, False, False
            self.reset()
        elif action == "undo": self.undo()
        elif action == "auto":
            self.autoplay = not self.autoplay
            if not self.thinking: self.maybe_start()
        elif action == "heat": self.heatmap = not self.heatmap
        elif action == "pause": self.paused = not self.paused
        elif action == "step" and self.thinking:
            self.paused = True
            self.search.step()
        elif action == "play": self.commit()
        elif action == "less": self.budget = max(50, self.budget//2)
        elif action == "more": self.budget = min(16000, self.budget*2)
        elif action == "root" and self.search:
            self.focus, self.page = self.search.root, 0
        elif action == "up" and self.focus and self.focus.parent:
            self.focus, self.page = self.focus.parent, 0
        elif action == "prev": self.page = max(0, self.page-1)
        elif action == "next": self.page += 1

    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            mapping = {pygame.K_r: "reset", pygame.K_f: "first", pygame.K_u: "undo", pygame.K_a: "auto", pygame.K_h: "heat", pygame.K_SPACE: "pause", pygame.K_s: "step", pygame.K_RETURN: "play", pygame.K_LEFT: "prev", pygame.K_RIGHT: "next", pygame.K_BACKSPACE: "up", pygame.K_HOME: "root"}
            self.action(mapping.get(event.key, ""))
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = ((event.pos[0]-self.offset[0])/self.scale, (event.pos[1]-self.offset[1])/self.scale)
            for rect, action in self.buttons:
                if rect.collidepoint(pos):
                    self.action(action)
                    return
            for rect, node in self.node_hits:
                if rect.collidepoint(pos):
                    self.focus, self.page = node, 0
                    return
            if not self.thinking and self.state.result() is None and self.state.player == self.human:
                x, y = pos
                if 62 <= x < 566 and 224 <= y < 728:
                    move = int((y-224)//84)*6 + int((x-62)//84)
                    if self.state.board[move] == 0:
                        self.play(move)

    def run(self):
        clock = pygame.time.Clock()
        self.draw()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                else:
                    self.handle(event)
            if running:
                self.update()
                self.draw()
                clock.tick(60)
        pygame.quit()


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="6×6 tic-tac-toe with a live PUCT search tree")
    parser.add_argument("--win-length", type=int, choices=range(3, 7), default=4)
    parser.add_argument("--simulations", type=positive, default=800, help="search simulations per move")
    parser.add_argument("--model", type=Path, help="optional local policy/value .npz checkpoint")
    parser.add_argument("--autoplay", action="store_true", help="watch AI versus AI")
    parser.add_argument("--train", action="store_true", help="train a local policy/value network by self-play")
    parser.add_argument("--games", type=positive, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.train:
            from .network import train
            args.model = args.model or Path("models/policy-value.npz")
            train(args)
        else:
            App(args).run()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"alphatic: {error}\n")


if __name__ == "__main__":
    main()
