"""Network shape / smoke tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dttt_train.network import (  # noqa: E402
    INPUT_CHANNELS,
    TOTAL_ACTIONS,
    DTTTNet,
    make_dummy_input,
)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_forward_shapes(batch_size: int) -> None:
    net = DTTTNet()
    net.eval()
    x = make_dummy_input(batch_size)
    assert x.shape == (batch_size, INPUT_CHANNELS, 4, 4)
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (batch_size, TOTAL_ACTIONS)
    assert value.shape == (batch_size, 1)
    # Value head is tanh-bounded.
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_forward_with_random_input() -> None:
    net = DTTTNet()
    net.eval()
    x = torch.randn((4, INPUT_CHANNELS, 4, 4))
    with torch.no_grad():
        logits, value = net(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_param_count_in_expected_range() -> None:
    """Sanity: ~316K params per the design doc (allow generous slack)."""
    net = DTTTNet()
    n = sum(p.numel() for p in net.parameters())
    # Doc estimate: ~316K. Allow 200K..500K to catch drastic architecture drift.
    assert 200_000 < n < 500_000, f"unexpected param count {n}"
