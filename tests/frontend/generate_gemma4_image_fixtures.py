# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate Gemma 4 image preprocessing fixtures from the canonical processor."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor


def pattern(width, height, seed):
    # Vectorized channel ramps exercise bicubic interpolation and extreme aspect clamping.
    y, x = np.indices((height, width))
    return np.stack(((x * 7 + seed * 31) % 256, (y * 11) % 256, (x + y * 3) % 256), -1).astype(np.uint8)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def generate(snapshot, output):
    raw = json.loads((snapshot / "processor_config.json").read_text())
    model = json.loads((snapshot / "config.json").read_text())
    cases = []
    specs = [
        (1, 1, 280),
        (91, 37, 280),
        (672, 672, 280),
        (1008, 672, 280),
        (1920, 1080, 280),
        (10, 3000, 280),
        (3000, 10, 280),
        (1920, 1080, 1120),
    ]
    for seed, (width, height, max_soft_tokens) in enumerate(specs):
        processor_kwargs = {**raw["image_processor"], "max_soft_tokens": max_soft_tokens}
        processor = Gemma4ImageProcessor(**processor_kwargs)
        pixels = pattern(width, height, seed)
        chw = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        result = processor(images=[chw], return_tensors="pt")
        positions = result["image_position_ids"][0]
        valid = (positions >= 0).all(-1)
        patches = int(valid.sum())
        valid_positions = positions[:patches]
        assert bool(valid[:patches].all()) and not bool(valid[patches:].any()), (
            f"unexpected non-contiguous padding in {width}x{height}"
        )
        grid_h = int(valid_positions[..., 1].max()) + 1
        grid_w = int(valid_positions[..., 0].max()) + 1
        pixel_values = result["pixel_values"][0, :patches]
        assert pixel_values.dtype == torch.float32
        assert valid_positions.dtype == torch.int64
        resized = processor.aspect_ratio_preserving_resize(
            chw,
            patch_size=processor.patch_size,
            max_patches=max_soft_tokens * processor.pooling_kernel_size * processor.pooling_kernel_size,
            pooling_kernel_size=processor.pooling_kernel_size,
            resample=tvF.InterpolationMode.BICUBIC,
        ).permute(1, 2, 0)
        assert resized.dtype == torch.uint8
        cases.append(
            dict(
                width=width,
                height=height,
                seed=seed,
                max_soft_tokens=max_soft_tokens,
                grid=[1, grid_h, grid_w],
                tokens=int(result["num_soft_tokens_per_image"][0]),
                patches=patches,
                pixels=digest(pixel_values),
                positions=digest(valid_positions),
                resized=digest(resized),
            )
        )
    profile = {
        key: model[key]
        for key in ("vision_config", "model_type", "image_token_id", "boi_token_id", "eoi_token_id")
        if key in model
    }
    output.write_text(json.dumps(dict(processor=raw, model=profile, cases=cases), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.snapshot, args.output)
