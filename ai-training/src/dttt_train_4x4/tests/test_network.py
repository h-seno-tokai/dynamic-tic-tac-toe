"""Network shape, ONNX export, and ORT-CPU parity tests for the 4x4 net."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from dttt_train_4x4.export_onnx import (  # noqa: E402
    INPUT_NAME,
    ONNX_OPSET,
    POLICY_NAME,
    WDL_NAME,
    export_to_onnx,
    parity_check,
)
from dttt_train_4x4.network import (  # noqa: E402
    INPUT_CHANNELS,
    NUM_RES_BLOCKS,
    TOTAL_ACTIONS,
    TRUNK_CHANNELS,
    WDL_OUTPUTS,
    DTTTNet4x4,
    make_dummy_input,
    wdl_to_scalar,
)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_forward_shapes(batch_size: int) -> None:
    net = DTTTNet4x4()
    net.eval()
    x = make_dummy_input(batch_size)
    assert x.shape == (batch_size, INPUT_CHANNELS, 4, 4)
    with torch.no_grad():
        logits, wdl = net(x)
    assert logits.shape == (batch_size, TOTAL_ACTIONS)
    assert wdl.shape == (batch_size, WDL_OUTPUTS)
    # WDL -> scalar Q in [-1, 1]
    q = wdl_to_scalar(wdl)
    assert torch.all(q >= -1.0)
    assert torch.all(q <= 1.0)


def test_forward_with_random_input() -> None:
    net = DTTTNet4x4()
    net.eval()
    x = torch.randn((4, INPUT_CHANNELS, 4, 4))
    with torch.no_grad():
        logits, wdl = net(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(wdl).all()


def test_param_count_in_expected_range() -> None:
    """Sanity: 96ch x 6 blocks - allow generous slack to catch drift.

    With 27 input channels and the architecture pinned to 96ch / 6 blocks the
    parameter count is ~0.7M; we keep generous bounds so a small head tweak
    does not break the test.
    """
    net = DTTTNet4x4()
    n = sum(p.numel() for p in net.parameters())
    assert 300_000 < n < 1_500_000, f"unexpected param count {n}"
    # Document the architecture pin so a width/depth change breaks the test.
    assert TRUNK_CHANNELS == 96
    assert NUM_RES_BLOCKS == 6


def test_onnx_export_and_parity(tmp_path: Path) -> None:
    """Export to ONNX at opset 17 and check PyTorch <-> ORT-CPU parity.

    Uses a small batch (1) since ``dynamic_axes=None``.
    """
    net = DTTTNet4x4()
    net.eval()

    out_path = tmp_path / "dttt_4x4.onnx"
    export_to_onnx(net, out_path)
    assert out_path.exists()

    # Confirm opset matches what we declared.
    model = onnx.load(str(out_path))
    opsets = {opset.domain: opset.version for opset in model.opset_import}
    # The default ONNX domain is the empty string.
    assert opsets.get("", 0) == ONNX_OPSET

    # Confirm I/O names line up with the export contract.
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    assert INPUT_NAME in input_names
    assert POLICY_NAME in output_names
    assert WDL_NAME in output_names

    # Cross-runtime numerical parity (PyTorch eval vs ONNX Runtime CPU).
    diff = parity_check(net, out_path, num_samples=4)
    assert diff < 1e-4, f"parity diff {diff:.3e} too large"
