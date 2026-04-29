"""AlphaZero universal network. Architecture per ``docs/07_ai_design.md`` 2.2.

Spec (must match exactly):

  Stem  : Conv3x3(27 -> 64) + BN + ReLU
  Body  : 4 x ResBlock
            Conv3x3(64 -> 64) + BN + ReLU
            Conv3x3(64 -> 64) + BN + skip + ReLU
  Heads :
    Place head : Conv1x1(64 -> 4)   -> flatten -> 64 logits
    Move head  : Conv1x1(64 -> 16)  -> flatten -> 256 logits
    -> concat to 320-d policy logits (NO mask in graph)
    Value head : Conv1x1(64 -> 32) + BN + ReLU
                 -> Global Average Pool
                 -> Linear(32 -> 64) + ReLU
                 -> Linear(64 -> 1) + tanh
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rules import MAX_BOARD, MOVE_ACTIONS, PLACE_ACTIONS, TOTAL_ACTIONS

# Architecture constants - DO NOT change without updating the design doc.
INPUT_CHANNELS: int = 27
TRUNK_CHANNELS: int = 128
NUM_RES_BLOCKS: int = 8
PLACE_HEAD_CHANNELS: int = 4   # 4 sizes
MOVE_HEAD_CHANNELS: int = 16   # 16 from-cells (4x4)
VALUE_HEAD_CHANNELS: int = 32
VALUE_HIDDEN: int = 64


class _ResBlock(nn.Module):
    def __init__(self, channels: int = TRUNK_CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class DTTTNet(nn.Module):
    """Universal AlphaZero network for the DTTT game.

    Forward signature::

        policy_logits, value = net(x)
        # x: (B, 27, 4, 4) float32
        # policy_logits: (B, 320)  -- raw logits, NO mask, NO softmax
        # value: (B, 1) in [-1, 1]  -- tanh
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        trunk_channels: int = TRUNK_CHANNELS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ) -> None:
        super().__init__()

        # Stem
        self.stem_conv = nn.Conv2d(
            input_channels, trunk_channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(trunk_channels)

        # Body
        self.res_blocks = nn.ModuleList(
            [_ResBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        # Policy heads
        self.place_head_conv = nn.Conv2d(
            trunk_channels, PLACE_HEAD_CHANNELS, kernel_size=1, bias=True
        )
        self.move_head_conv = nn.Conv2d(
            trunk_channels, MOVE_HEAD_CHANNELS, kernel_size=1, bias=True
        )

        # Value head
        self.value_conv = nn.Conv2d(
            trunk_channels, VALUE_HEAD_CHANNELS, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Stem
        h = F.relu(self.stem_bn(self.stem_conv(x)))

        # Body
        for block in self.res_blocks:
            h = block(h)

        # Policy: place (B, 4, 4, 4) -> (B, 64); move (B, 16, 4, 4) -> (B, 256)
        place = self.place_head_conv(h)
        # flatten as (sizes, rows, cols) -> 64
        place_logits = place.flatten(start_dim=1)
        assert place_logits.shape[1] == PLACE_ACTIONS, (
            f"place head produced {place_logits.shape[1]} logits, expected {PLACE_ACTIONS}"
        )

        move = self.move_head_conv(h)
        # The move head's 16 channels correspond to from-cell index, and the
        # 4x4 spatial dims to to-cell. We flatten as (to_row, to_col, from_idx)
        # so that flatten ordering matches encoding.action_index_to_move.
        # PyTorch flatten is C-order, so the layout becomes
        # (channel=from_idx, row=to_row, col=to_col).
        # We permute to (B, to_row, to_col, from_idx) before flatten:
        move_perm = move.permute(0, 2, 3, 1).contiguous()
        move_logits = move_perm.flatten(start_dim=1)
        assert move_logits.shape[1] == MOVE_ACTIONS, (
            f"move head produced {move_logits.shape[1]} logits, expected {MOVE_ACTIONS}"
        )

        policy_logits = torch.cat([place_logits, move_logits], dim=1)
        assert policy_logits.shape[1] == TOTAL_ACTIONS

        # Value
        v = F.relu(self.value_bn(self.value_conv(h)))
        # Global average pool over the 4x4 spatial dims.
        v = v.mean(dim=(2, 3))
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return policy_logits, v


def make_dummy_input(batch_size: int = 1) -> torch.Tensor:
    """Build a zero input tensor of the canonical shape ``(B, 27, 4, 4)``."""
    return torch.zeros((batch_size, INPUT_CHANNELS, MAX_BOARD, MAX_BOARD), dtype=torch.float32)


__all__ = [
    "DTTTNet",
    "make_dummy_input",
    "INPUT_CHANNELS",
    "TRUNK_CHANNELS",
    "NUM_RES_BLOCKS",
    "PLACE_ACTIONS",
    "MOVE_ACTIONS",
    "TOTAL_ACTIONS",
]
