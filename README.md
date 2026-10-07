# AlphaTic

A Pygame 6×6 tic-tac-toe game with a live, interactive AI search tree. Connect **four** horizontally, vertically, or diagonally to win by default; choose three, five, or six with `--win-length`.

## Run

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run alphatic
```

Alternatively, install with `python -m pip install -e .` and run `alphatic` or `python -m alphatic`. The `pygame-ce` dependency provides the `pygame` module. The game runs locally without an API key, network service, or GPU.

```sh
uv run alphatic --win-length 6 --simulations 1500
uv run alphatic --autoplay
```

You are X and move first. Click an empty square to play. The window can be resized; its contents scale to fit.

## Controls and tree inspector

| Control | Action |
| --- | --- |
| R / New game | Start over as X, with autoplay and pause off |
| F / AI first | Start over as O; the AI plays X |
| U / Undo | Rewind to your previous turn and stop autoplay |
| A / Auto | Toggle AI vs AI; an already-running search finishes its move |
| Space | Pause/resume search and automatic moves |
| S / Step | Pause and run exactly one search simulation |
| Enter / Play | Play the current best AI move immediately |
| H / Heat | Toggle search visit counts on the main board |
| − / + buttons | Halve/double the simulations allowed per move |
| Click a tree node | Explore that position's children |
| Backspace / Up | Return to the parent position |
| Home / Root | Return to the search root |
| Left / Right | Page through branches |
| Escape | Quit |

The inspector shows the actual search nodes, including miniature boards. It displays five children per page, sorted by visits, and up to two replies under each child. Every expanded branch is accessible by paging and clicking. An unexpanded node has no continuations yet. The last search stays visible after the AI moves, until another search starts.

- **N**: number of simulations that visited the node.
- **P**: policy prior used to guide exploration from the parent.
- **Q**: mean estimated outcome, consistently shown from the **search root player's** perspective. −1 means loss, +1 means win, and 0 is neutral. These are estimates, not calibrated win probabilities.
- **Gold border**: nodes in the most recent simulation's path. On the main board, gold marks the last move.
- **Heatmap**: brighter empty cells received more search visits. It appears only while the displayed board matches the search root.

Search runs in small time slices so the UI can remain interactive. Automatic moves wait at least 1.2 seconds to make the tree visible; pause to inspect it longer. Step can exceed the configured budget intentionally.

## How the AI works

The search is inspired by [AlphaGo/AlphaZero's combination of policy, value, and Monte Carlo tree search](https://deepmind.google/research/alphago/). It uses **PUCT** to balance high-value moves with exploration of promising policy priors:

```text
score(child) = -Q(child) + c × P(child) × sqrt(N(parent)) / (1 + N(child))
```

Internally, values belong to each node's player to move, so the search negates the child's value during selection and flips the sign at every backup. Each simulation descends the tree, evaluates and expands one leaf (or uses the exact terminal result), and backs up the value. The most visited root move is played. All legal moves remain represented in the tree. A fresh tree is built for each AI turn.

**The default AI uses a neural network for both position evaluation and move priors.** Every nonterminal search leaf is evaluated by the network's value head; there is no heuristic blend or tactical override. Terminal wins, losses, and draws still use exact game results. The standard four-in-a-row game loads a bundled starter checkpoint trained for 300 self-play games with 128 search simulations per move, starting from random weights. The UI shows the loaded model's training game count.

The NumPy network has three relative board planes (own stones, opposing stones, empty cells), a shared 64-unit tanh layer, a masked 36-move policy head, and a tanh value head. Training uses search visit distributions and final game outcomes, with eight board symmetries, root Dirichlet noise, early-game move sampling, a bounded replay buffer, and minibatch gradient descent. Self-play uses the network from the very first game.

## Train a policy/value network

```sh
uv run alphatic --train --games 100 --simulations 64 --model models/policy-value.npz
uv run alphatic --model models/policy-value.npz
```

To continue training the bundled starter, copy it to your own checkpoint first:

```sh
mkdir -p models
cp src/alphatic/checkpoints/connect-4.npz models/policy-value.npz
uv run alphatic --train --games 100 --simulations 128 --model models/policy-value.npz
uv run alphatic --model models/policy-value.npz
```

The checkpoint is saved after every game. Running training again with the same path resumes its weights and total game count; the in-memory replay buffer starts fresh. Use `--seed` for reproducible fresh runs. Models store their win length: train and play with the same `--win-length`, and use separate files for different rules.

The starter is a modest demonstration model; **its playing strength has not been benchmarked**. Other win lengths start with a fresh neural network labeled `untrained` unless you supply a matching trained `--model`. There is no automatic heuristic fallback. Train a separate model for those rules, for example:

```sh
uv run alphatic --train --win-length 6 --games 100 --simulations 128 --model models/connect-6.npz
uv run alphatic --win-length 6 --model models/connect-6.npz
```

Training time depends on the number of games and simulations. The historical tactical evaluator remains in the source for baseline tests, but the game and self-play trainer do not use it.

## Verify

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover all winning windows for supported rule lengths, draw/illegal-move handling, tactical wins and blocks for both players, search backup signs and visit counts, complete self-play, neural masking/learning/checkpoint round-trips, symmetry augmentation, and Pygame interactions using SDL's headless driver.

## Code

- `src/alphatic/game.py`: immutable board and connect-k rules.
- `src/alphatic/search.py`: PUCT nodes, search, and tactical evaluator.
- `src/alphatic/network.py`: policy/value network and self-play trainer.
- `src/alphatic/app.py`: Pygame UI and command-line options.
