# dttt-train

AlphaZero training scaffold for the DTTT Gobblet-clone game.

Implements the **universal network** described in `docs/07_ai_design.md`:
a single 4x4x27-input ResNet (4 blocks, 64 channels) that covers both the
**3x3 classic** and **4x4 huge** presets, with a 320-d policy head and a
scalar tanh value head.

## Setup (uv recommended)

```bash
cd ai-training
uv venv
uv sync --extra dev
# or, with pip:
# python -m venv .venv && source .venv/bin/activate
# pip install -e ".[dev]"
```

Optional experiment tracking (not required):

```bash
uv pip install tensorboard   # or wandb
```

## Run

```bash
# Run the (skeleton) training loop
uv run python -m dttt_train.train --total-steps 100 --num-workers 2

# Export a checkpoint to ONNX and run a 2-way parity test (PyTorch <-> ORT CPU)
uv run python -m dttt_train.export_onnx --ckpt runs/latest.pt --out model.onnx

# Tests
uv run pytest
uv run ruff check .
uv run mypy
```

## ONNX export pitfalls (see `docs/10_risks.md` 1.2)

These are enforced in `export_onnx.py` and must not be relaxed:

1. **opset 17** fixed (best balance for ONNX Runtime Web).
2. `model.eval()` is called before `torch.onnx.export` so that BN folds the
   training statistics. Exporting in train mode breaks browser inference.
3. **`dynamic_axes=None`** - input shape is fixed at `(1, 27, 4, 4)`. This
   keeps the graph friendly to int8 quantisation later.
4. **No legal-action mask in the graph.** A `-inf + softmax` path inside the
   ONNX graph yields NaN under ORT Web. The mask is applied in Python /
   TypeScript MCTS code, never in the network forward.
5. **3-way parity test** - PyTorch eval / ORT CPU / ORT Web outputs must
   agree to `max abs diff < 1e-4`. The Python side covers the first two; the
   TypeScript app covers the third.

## Layout

```
src/dttt_train/
  rules.py          dataclass mirror of the TS GameRules / Move types
  engine.py         pure-Python rule engine (parity with TS)
  encoding.py       state -> (27,4,4) tensor, legal_action_mask -> (320,)
  network.py        ResNet (stem + 4 blocks + policy/value heads)
  mcts.py           leaf-batched MCTS w/ virtual loss, mask in Python
  selfplay.py       play_game(...) -> training tuples
  replay_buffer.py  FIFO buffer
  train.py          training loop skeleton (multiprocessing self-play)
  export_onnx.py    checkpoint -> ONNX with parity test
tests/              engine / network / encoding tests
```

This is a **scaffold**: training-loop bodies are intentionally lightweight.
The network architecture and the 27-channel encoding are real and runnable.
