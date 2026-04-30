"""Checkpoint -> ONNX exporter with parity check.

Following ``docs/10_risks.md`` 1.2 strictly:

* opset 17 fixed
* model.eval() called before export (BN folds training stats)
* dynamic_axes=None - input shape (1, 27, 4, 4) is fixed
* No legal-action mask in the graph (mask is applied in MCTS / TS code)
* Asserts max abs diff < 1e-4 between PyTorch (eval) and ONNX Runtime CPU.
  (The third leg, ONNX Runtime Web, is checked from the TS side.)

Outputs:
  * ``policy_logits`` : (1, 320)  raw policy logits (apply mask + softmax in TS)
  * ``wdl_logits``    : (1, 3)    raw WDL logits  (Win, Draw, Loss)
                                  TS computes scalar Q = softmax then
                                  P(win) - P(loss) for MCTS backups.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .network import INPUT_CHANNELS, DTTTNet
from .rules import MAX_BOARD

logger = logging.getLogger(__name__)

ONNX_OPSET: int = 17
INPUT_NAME: str = "input"
POLICY_NAME: str = "policy_logits"
WDL_NAME: str = "wdl_logits"
PARITY_TOLERANCE: float = 1e-4


def _load_checkpoint(ckpt_path: Path, prefer_ema: bool = True) -> DTTTNet:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if prefer_ema and isinstance(blob, dict) and "ema_state" in blob:
        state = blob["ema_state"]
        logger.info("loaded EMA weights from %s", ckpt_path)
    else:
        state = blob.get("model_state", blob)
    net = DTTTNet()
    net.load_state_dict(state)
    net.eval()
    return net


def export_to_onnx(net: DTTTNet, out_path: Path) -> None:
    net.eval()
    dummy = torch.zeros((1, INPUT_CHANNELS, MAX_BOARD, MAX_BOARD), dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(out_path),
        input_names=[INPUT_NAME],
        output_names=[POLICY_NAME, WDL_NAME],
        opset_version=ONNX_OPSET,
        dynamic_axes=None,
        do_constant_folding=True,
        export_params=True,
    )
    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)
    logger.info("exported and validated ONNX -> %s", out_path)


def parity_check(net: DTTTNet, onnx_path: Path, num_samples: int = 4) -> float:
    net.eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for i in range(num_samples):
        x_np = rng.standard_normal(
            (1, INPUT_CHANNELS, MAX_BOARD, MAX_BOARD)
        ).astype(np.float32)
        with torch.no_grad():
            torch_logits, torch_wdl = net(torch.from_numpy(x_np))
        torch_logits_np = torch_logits.cpu().numpy()
        torch_wdl_np = torch_wdl.cpu().numpy()

        ort_outputs = sess.run([POLICY_NAME, WDL_NAME], {INPUT_NAME: x_np})
        ort_logits, ort_wdl = ort_outputs

        d_logits = float(np.max(np.abs(torch_logits_np - ort_logits)))
        d_wdl = float(np.max(np.abs(torch_wdl_np - ort_wdl)))
        sample_diff = max(d_logits, d_wdl)
        logger.info(
            "parity sample %d: dlogits=%.3e dwdl=%.3e", i, d_logits, d_wdl
        )
        max_diff = max(max_diff, sample_diff)

    assert max_diff < PARITY_TOLERANCE, (
        f"ONNX parity check failed: max abs diff {max_diff:.3e} >= {PARITY_TOLERANCE:.0e}"
    )
    return max_diff


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export DTTT checkpoint to ONNX with parity check")
    parser.add_argument("--ckpt", type=str, required=True, help="path to .pt checkpoint")
    parser.add_argument("--out", type=str, default="model.onnx", help="output ONNX path")
    parser.add_argument("--samples", type=int, default=4, help="parity-check sample count")
    parser.add_argument("--no-ema", action="store_true",
                        help="export raw model weights instead of EMA")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    net = _load_checkpoint(Path(args.ckpt), prefer_ema=not args.no_ema)
    out_path = Path(args.out)
    export_to_onnx(net, out_path)
    diff = parity_check(net, out_path, num_samples=args.samples)
    logger.info("parity OK, max abs diff = %.3e", diff)


if __name__ == "__main__":
    main()
