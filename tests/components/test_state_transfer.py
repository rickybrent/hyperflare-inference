# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Device-free validation checks for PreparedStateTransfer."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ignition.components.sampling import DeviceRequestState
from ignition.components.state import StateLayout
from ignition.components.state_transfer import PreparedStateTransfer, StateCommands


def test_prepared_state_transfer_rejects_row_bytes_over_int32():
    """row_bytes above INT32_MAX - 65_535 * 4096 is rejected before the i32 stride induction can wrap."""
    # First 256-aligned row_bytes > (2^31 - 1) - 65_535 * 4096 = 1,879,052,287 B.
    layout = StateLayout.from_specs([
        ("payload", (1_879_051_776,), torch.uint8),
        ("target_page", (128,), torch.uint8),
        ("consumed_tokens", (1,), torch.int64),
    ])
    assert layout.row_bytes == 1_879_052_288
    with FakeTensorMode():
        device = torch.device("cuda:0")
        pool = SimpleNamespace(
            layout=layout,
            backing=torch.empty((6, layout.row_bytes), dtype=torch.uint8, device=device),
        )
        tables = torch.zeros((4, 2), dtype=torch.int32, device=device)
        pages = torch.empty((4, 128), dtype=torch.uint8, device=device)
        with pytest.raises(ValueError, match="strided copy offset range"):
            PreparedStateTransfer(
                pool,
                StateCommands.allocate(device, capacity=4),
                tables,
                pages=[("target_page", pages)],
                page_tokens=256,
                state_rows=torch.zeros((4, 8), dtype=torch.int64, device=device),
                requests=DeviceRequestState.allocate(device, vocab_size=128, capacity=4),
            )
