"""Small NumPy policy/value network and local self-play training."""
from pathlib import Path
import numpy as np
from .game import State
from .search import Search


def encode(state: State) -> np.ndarray:
    board = np.asarray(state.board) * state.player
    return np.concatenate([board == 1, board == -1, board == 0]).astype(np.float32)


class Network:
    @property
    def name(self):
        training = f"{self.games} games" if self.games else "untrained"
        return f"Neural policy/value · {training}"

    def __init__(self, k: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.games = 0
        self.w1 = rng.normal(0, .1, (108, 64)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.wp = rng.normal(0, .1, (64, 36)).astype(np.float32)
        self.bp = np.zeros(36, dtype=np.float32)
        self.wv = rng.normal(0, .1, (64, 1)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.wp + self.bp
        logits = np.where(x[..., 72:] > 0, logits, -1e9)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        v = np.tanh(h @ self.wv + self.bv)
        return h, p, v

    def evaluate(self, state: State):
        if state.k != self.k:
            raise ValueError("model win length does not match game")
        _, p, v = self.forward(encode(state))
        return {m: float(p[m]) for m in state.legal_moves()}, float(v[0])

    def fit(self, x, target_p, target_v, lr=.02):
        h, p, v = self.forward(x)
        n = len(x)
        dp = (p - target_p) / n
        dv = 2 * (v - target_v) * (1 - v * v) / n
        dh = (dp @ self.wp.T + dv @ self.wv.T) * (1 - h * h)
        gradients = (x.T @ dh, dh.sum(0), h.T @ dp, dp.sum(0), h.T @ dv, dv.sum(0))
        for name, grad in zip(("w1", "b1", "wp", "bp", "wv", "bv"), gradients):
            param = getattr(self, name)
            param -= lr * (grad + .0001 * param)
        return float(-np.mean(np.sum(target_p * np.log(p + 1e-9), axis=1)) + np.mean((v - target_v) ** 2))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            np.savez_compressed(file, k=self.k, games=self.games, **{key: getattr(self, key) for key in ("w1", "b1", "wp", "bp", "wv", "bv")})

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(int(data["k"]))
            model.games = int(data["games"])
            for key in ("w1", "b1", "wp", "bp", "wv", "bv"):
                array = data[key]
                if array.shape != getattr(model, key).shape or not np.isfinite(array).all():
                    raise ValueError(f"invalid model parameter: {key}")
                setattr(model, key, array.astype(np.float32))
        return model


def default_network(k: int = 4) -> Network:
    """Load the packaged starter, or clearly label a fresh network for other rules."""
    path = Path(__file__).with_name("checkpoints") / f"connect-{k}.npz"
    model = Network.load(path) if path.exists() else Network(k)
    if model.k != k:
        raise ValueError("starter model win length does not match game")
    return model


def augment(state, policy):
    planes = encode(state).reshape(3, 6, 6)
    target = policy.reshape(6, 6)
    for turns in range(4):
        x = np.rot90(planes, turns, axes=(1, 2))
        p = np.rot90(target, turns)
        yield x.flatten(), p.flatten()
        yield x[:, :, ::-1].flatten(), p[:, ::-1].flatten()


def train(args):
    rng = np.random.default_rng(args.seed)
    model = Network.load(args.model) if args.model.exists() else Network(args.win_length, args.seed)
    if model.k != args.win_length:
        raise ValueError("existing model uses a different win length; use a different --model path")
    replay = []
    for game in range(args.games):
        state = State(k=args.win_length)
        trajectory = []
        while state.result() is None:
            search = Search(state, model)
            search.step()
            children = list(search.root.children.values())
            noise = rng.dirichlet(np.full(len(children), .3))
            for child, sample in zip(children, noise):
                child.prior = .75 * child.prior + .25 * float(sample)
            search.run(args.simulations)
            policy = np.zeros(36, dtype=np.float32)
            for move, child in search.root.children.items():
                policy[move] = child.visits
            policy /= policy.sum()
            trajectory.append((state, policy))
            move = int(rng.choice(36, p=policy)) if len(trajectory) <= 10 else search.best_move()
            state = state.play(move)
        winner = state.result()
        for position, policy in trajectory:
            for x, p in augment(position, policy):
                replay.append((x, p, float(winner * position.player)))
        replay = replay[-16000:]
        loss = 0.0
        for _ in range(20):
            batch = [replay[i] for i in rng.integers(len(replay), size=min(64, len(replay)))]
            x, p, v = zip(*batch)
            loss = model.fit(np.asarray(x), np.asarray(p), np.asarray(v, dtype=np.float32)[:, None])
        model.games += 1
        model.save(args.model)
        print(f"Game {game + 1}/{args.games} | total {model.games} | moves {len(trajectory)} | loss {loss:.3f} | saved {args.model}", flush=True)
