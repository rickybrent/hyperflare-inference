# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-copy grid striding preserves (row, tile) direct launches and supports rows over 256 MiB - 4 KiB."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState
from ignition.components.state import StateLayout, StateStorage
from ignition.components.state_transfer import (
    _COPY_BLOCK,
    _MAX_GRID_Y,
    PreparedStateTransfer,
    StateCommands,
    _copy_rows,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)


def _expected_copy_and_reset(source_row, sentinel, excluded):
    """Build expected copy and reset rows honoring excluded byte spans."""
    copied = source_row.clone()
    reset = torch.zeros_like(source_row)
    for start, end in excluded:
        copied[start:end].fill_(sentinel)
        reset[start:end].fill_(sentinel)
    return copied, reset


def _replay_and_verify(
    graph,
    backing,
    src_row,
    copy_row,
    reset_row,
    excluded,
    *,
    prepare_replay=None,
    check_after=None,
    untouched_row=None,
):
    """Dirty backing buffers before each CUDA graph replay and assert byte-for-byte results."""
    for src_val, sentinel in ((17, 231), (53, 199), (121, 77)):
        backing.fill_(sentinel)
        backing[src_row].fill_(src_val)
        backing[src_row, :64].fill_(src_val + 1)
        backing[src_row, -64:].fill_(src_val + 2)
        if prepare_replay is not None:
            prepare_replay()
        expected_src = backing[src_row].cpu().clone()
        graph.replay()
        torch.cuda.synchronize()
        if check_after is not None:
            check_after()
        expected_copy, expected_reset = _expected_copy_and_reset(expected_src, sentinel, excluded)
        if untouched_row is not None:
            assert torch.equal(backing[untouched_row].cpu(), torch.full((backing.shape[1],), sentinel, dtype=torch.uint8))
        assert torch.equal(backing[src_row].cpu(), expected_src)
        assert torch.equal(backing[copy_row].cpu(), expected_copy)
        assert torch.equal(backing[reset_row].cpu(), expected_reset)


@pytest.mark.parametrize("grid_y", [8, 3])
@torch.no_grad()
def test_copy_rows_preserves_row_tile_grid_and_excluded_spans(grid_y):
    device = torch.device("cuda:0")
    row_bytes = 8 * _COPY_BLOCK
    # First excluded span is deliberately non-tile-aligned to verify byte-exact masking.
    excluded = ((3 * 1024, 5 * 1024), (4 * _COPY_BLOCK, 5 * _COPY_BLOCK))
    backing = torch.full((4, row_bytes), 231, dtype=torch.uint8, device=device)
    source = torch.tensor([1, -1], dtype=torch.int64, device=device)
    destination = torch.tensor([2, 3], dtype=torch.int64, device=device)
    operation = torch.tensor([0, 1], dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    args = (backing, source, destination, operation, count, errors, row_bytes, _COPY_BLOCK, excluded)

    _copy_rows[(2, grid_y)](*args)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _copy_rows[(2, grid_y)](*args)
        _replay_and_verify(graph, backing, 1, 2, 3, excluded, untouched_row=0)
    finally:
        graph.reset()


@pytest.mark.parametrize(("total_tiles", "expected_y"), [(8, 8), (65_535, 65_535), (_MAX_GRID_Y + 2, 65_535)])
@torch.no_grad()
def test_prepared_state_transfer_caps_grid_y_and_replays_rows(total_tiles, expected_y):
    """PreparedStateTransfer.prepare() keeps unstrided grids <= _MAX_GRID_Y tiles and caps larger rows."""
    device = torch.device("cuda:0")
    capacity = 4
    page_tokens = 256
    body_tiles = total_tiles - 4
    layout = StateLayout.from_specs([
        ("payload_head", (_COPY_BLOCK,), torch.uint8),
        ("excluded_head", (_COPY_BLOCK,), torch.uint8),
        ("payload_body", (body_tiles * _COPY_BLOCK,), torch.uint8),
        ("excluded_tail", (_COPY_BLOCK,), torch.uint8),
        ("target_page", (128,), torch.uint8),
        ("consumed_tokens", (1,), torch.int64),
        ("payload_tail", (_COPY_BLOCK - 512,), torch.uint8),
    ])
    assert layout.row_bytes == total_tiles * _COPY_BLOCK

    # Hard-code expected excluded spans from the fixed tile layout rather than querying layout.region().
    excluded = (
        (_COPY_BLOCK, 2 * _COPY_BLOCK),
        ((2 + body_tiles) * _COPY_BLOCK, (3 + body_tiles) * _COPY_BLOCK),
    )

    pool = StateStorage.allocate(layout, slots=capacity + 2, device=device)
    commands = StateCommands.allocate(device, capacity=capacity)
    tables = torch.zeros((capacity, 2), dtype=torch.int32, device=device)
    pages = torch.zeros((4, 128), dtype=torch.uint8, device=device)
    state_rows = torch.zeros((capacity, 8), dtype=torch.int64, device=device)
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=capacity)

    transfer = PreparedStateTransfer(
        pool,
        commands,
        tables,
        pages=[("target_page", pages)],
        page_tokens=page_tokens,
        state_rows=state_rows,
        requests=requests,
        excluded_regions=("excluded_head", "excluded_tail"),
    )
    transfer.prepare()
    # PreparedStateTransfer exposes no public grid accessor; observe the prepared row-copy launch via _copies[0].
    assert transfer._copies[0].grid == (capacity, expected_y, 1)

    state_rows[0, 0] = 1
    state_rows[1, 0] = 3
    requests.consumed_tokens[0] = page_tokens
    commands.request_slots[:2].copy_(torch.tensor([0, 1], dtype=torch.int64, device=device))
    commands.source_slots[:2].copy_(torch.tensor([1, -1], dtype=torch.int64, device=device))
    commands.destination_slots[:2].copy_(torch.tensor([2, 3], dtype=torch.int64, device=device))
    commands.operation[:2].copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
    commands.expected_consumed[:2].copy_(torch.tensor([page_tokens, 0], dtype=torch.int64, device=device))
    commands.num_ops.fill_(2)

    def prepare_replay():
        commands.errors.fill_(99)

    def check_after():
        assert (commands.errors == 0).all()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            transfer.validate()
            transfer.apply()
        _replay_and_verify(
            graph,
            pool.backing,
            1,
            2,
            3,
            excluded,
            prepare_replay=prepare_replay,
            check_after=check_after,
        )
    finally:
        graph.reset()
