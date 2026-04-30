"""Network shape / smoke tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dttt_train.network import (  # noqa: E402
    INPUT_CHANNELS,
    TOTAL_ACTIONS,
    WDL_OUTPUTS,
    DTTTNet,
    make_dummy_input,
    wdl_to_scalar,
)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_forward_shapes(batch_size: int) -> None:
    net = DTTTNet()
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
    net = DTTTNet()
    net.eval()
    x = torch.randn((4, INPUT_CHANNELS, 4, 4))
    with torch.no_grad():
        logits, wdl = net(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(wdl).all()


def test_param_count_in_expected_range() -> None:
    """Sanity: 128ch x 8 blocks - allow generous slack to catch drift."""
    net = DTTTNet()
    n = sum(p.numel() for p in net.parameters())
    assert 1_000_000 < n < 3_000_000, f"unexpected param count {n}"
