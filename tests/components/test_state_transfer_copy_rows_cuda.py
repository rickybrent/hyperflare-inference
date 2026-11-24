# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-copy grid striding preserves (row, tile) launches and supports rows over 256 MiB."""

import os

import pytest
import torch
from triton.runtime.interpreter import InterpretedFunction

from ignition.components.state_transfer import _copy_rows


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                reason="requires an explicitly assigned CUDA device",
            ),
        ),
    ],
)
@pytest.mark.parametrize("grid_y", [8, 3])
def test_copy_rows_preserves_row_tile_grid_and_excluded_spans(device, grid_y):
    backing = torch.full((4, 8192), 231, dtype=torch.uint8, device=device)
    backing[1].fill_(17)
    source = torch.tensor([1, -1], device=device)
    destination = torch.tensor([2, 3], device=device)
    operation = torch.tensor([0, 1], dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    args = (backing, source, destination, operation, count, errors, 8192, 1024, ((768, 1792), (4096, 5120)))
    kernel = _copy_rows if device == "cuda" else InterpretedFunction(_copy_rows.fn)
    kernel[(2, grid_y)](*args)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                kernel[(2, grid_y)](*args)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
        finally:
            graph.reset()
    wanted = torch.full((4, 8192), 231, dtype=torch.uint8)
    wanted[1:3].fill_(17)
    wanted[3].zero_()
    for start, end in args[-1]:
        wanted[2:, start:end].fill_(231)
    assert torch.equal(backing.cpu(), wanted)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")
@torch.no_grad()
def test_copy_rows_exceeds_65535_tile_row_size():
    """Rows exceeding 65,535 4 KiB tiles (>256 MiB) copy and reset cleanly under CUDA's gridDim.y limit."""
    device = torch.device("cuda:0")
    block = 4096
    tiles = 65_537
    row_bytes = tiles * block
    backing = torch.full((3, row_bytes), 231, dtype=torch.uint8, device=device)
    backing[0].fill_(17)
    backing[0, :64].fill_(42)
    backing[0, -64:].fill_(99)
    source = torch.tensor([0, -1], dtype=torch.int64, device=device)
    destination = torch.tensor([1, 2], dtype=torch.int64, device=device)
    operation = torch.tensor([0, 1], dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    excluded = ((block, 2 * block), (row_bytes - 2 * block, row_bytes - block))
    args = (backing, source, destination, operation, count, errors, row_bytes, block, excluded)

    grid = (2, min(tiles, 32768))
    _copy_rows[grid](*args)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _copy_rows[grid](*args)
        graph.replay()
        torch.cuda.synchronize()
    finally:
        graph.reset()

    assert (backing[1, :64] == 42).all()
    assert (backing[1, 64:block] == 17).all()
    assert (backing[1, block : 2 * block] == 231).all()
    assert (backing[1, 2 * block : row_bytes - 2 * block] == 17).all()
    assert (backing[1, row_bytes - 2 * block : row_bytes - block] == 231).all()
    assert (backing[1, row_bytes - block : -64] == 17).all()
    assert (backing[1, -64:] == 99).all()

    assert (backing[2, :block] == 0).all()
    assert (backing[2, block : 2 * block] == 231).all()
    assert (backing[2, 2 * block : row_bytes - 2 * block] == 0).all()
    assert (backing[2, row_bytes - 2 * block : row_bytes - block] == 231).all()
    assert (backing[2, row_bytes - block :] == 0).all()
