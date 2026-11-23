# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate Gemma 4 frontend image-profile extraction and negative gates."""

import copy
import json
import os
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from ignition.preparation.frontend import image_settings, prepare_frontend


def _write_synthetic_snapshot(
    root: Path,
    *,
    processor=None,
    model=None,
    tokenizer_config=None,
    vocab=None,
):
    root.mkdir(parents=True, exist_ok=True)
    vocab = vocab or {
        "<unk>": 0,
        "extra": 1,
        "<|image|> extra": 2,
        "<|image>": 255999,
        "<|image|>": 258880,
        "<image|>": 258882,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(root / "tokenizer.json"))

    processor = processor or {
        "processor_class": "Gemma4Processor",
        "image_processor": {
            "image_processor_type": "Gemma4ImageProcessor",
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": False,
            "do_convert_rgb": True,
            "resample": 3,
            "rescale_factor": 1 / 255,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
        },
    }
    model = model or {
        "model_type": "gemma4",
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "vision_config": {
            "model_type": "gemma4_vision",
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "num_hidden_layers": 16,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "head_dim": 64,
            "intermediate_size": 3072,
        },
    }
    tokenizer_config = tokenizer_config or {
        "image_token": "<|image|>",
        "boi_token": "<|image>",
        "eoi_token": "<image|>",
    }
    (root / "processor_config.json").write_text(json.dumps(processor))
    (root / "config.json").write_text(json.dumps(model))
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return processor, model, tokenizer_config


def test_prepare_frontend_accepts_e2b_sized_gemma4_tower(tmp_path):
    _write_synthetic_snapshot(tmp_path)
    prepared = prepare_frontend(
        tmp_path,
        eos_token_ids=[1],
        image_profile="gemma4",
        preprocessor=tmp_path / "processor_config.json",
    )
    assert prepared["image"] == {
        "gemma4": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "image_token": "<|image|>",
            "start_token": "<|image>",
            "end_token": "<image|>",
        }
    }


def test_image_settings_accepts_object_tokens_and_omitted_model_ids(tmp_path):
    processor, model, tokenizer_config = _write_synthetic_snapshot(
        tmp_path,
        tokenizer_config={
            "image_token": {"content": "<|image|>"},
            "boi_token": {"content": "<|image>"},
            "eoi_token": {"content": "<image|>"},
        },
    )
    model_without_ids = {
        k: v for k, v in model.items() if k not in ("image_token_id", "boi_token_id", "eoi_token_id")
    }
    resolved = image_settings(
        processor,
        model_without_ids,
        "gemma4",
        tokenizer_config=tokenizer_config,
        snapshot=tmp_path,
    )
    assert resolved["gemma4"]["image_token"] == "<|image|>"
    assert resolved["gemma4"]["start_token"] == "<|image>"
    assert resolved["gemma4"]["end_token"] == "<image|>"


def test_image_settings_rejects_gemma4_unified_and_bad_geometry(tmp_path):
    processor, model, tokenizer_config = _write_synthetic_snapshot(tmp_path)

    # Gemma 4 12B uses gemma4_unified / Gemma4UnifiedImageProcessor, which must be rejected.
    unified_model = copy.deepcopy(model)
    unified_model["model_type"] = "gemma4_unified"
    with pytest.raises(ValueError, match="model_type"):
        image_settings(processor, unified_model, "gemma4", tokenizer_config=tokenizer_config, snapshot=tmp_path)

    unified_vision = copy.deepcopy(model)
    unified_vision["vision_config"]["model_type"] = "gemma4_unified_vision"
    with pytest.raises(ValueError, match="model_type"):
        image_settings(processor, unified_vision, "gemma4", tokenizer_config=tokenizer_config, snapshot=tmp_path)

    unified_proc = copy.deepcopy(processor)
    unified_proc["image_processor"]["image_processor_type"] = "Gemma4UnifiedImageProcessor"
    with pytest.raises(ValueError, match="image_processor_type"):
        image_settings(unified_proc, model, "gemma4", tokenizer_config=tokenizer_config, snapshot=tmp_path)

    # Non-dict guards and bad numeric geometry / bool guards.
    with pytest.raises(ValueError):
        image_settings({"processor_class": "Gemma4Processor", "image_processor": None}, model, "gemma4")
    with pytest.raises(ValueError):
        image_settings(processor, {**model, "vision_config": None}, "gemma4")
    for field, bad_value in (("patch_size", 14), ("pooling_kernel_size", 2), ("max_soft_tokens", 300), ("max_soft_tokens", True)):
        bad_proc = copy.deepcopy(processor)
        bad_proc["image_processor"][field] = bad_value
        with pytest.raises(ValueError):
            image_settings(bad_proc, model, "gemma4", tokenizer_config=tokenizer_config, snapshot=tmp_path)


def test_image_settings_fails_closed_without_tokenizer_or_bad_envelope_tokens(tmp_path):
    processor, model, tokenizer_config = _write_synthetic_snapshot(tmp_path)

    # Failing closed when tokenizer_config or snapshot is omitted.
    with pytest.raises(ValueError, match="tokenizer_config and snapshot"):
        image_settings(processor, model, "gemma4")
    with pytest.raises(ValueError, match="tokenizer_config and snapshot"):
        image_settings(processor, model, "gemma4", tokenizer_config=tokenizer_config)

    # Missing, empty, and duplicate envelope tokens.
    with pytest.raises(ValueError, match="boi_token"):
        image_settings(processor, model, "gemma4", tokenizer_config={"image_token": "<|image|>", "eoi_token": "<image|>"}, snapshot=tmp_path)
    with pytest.raises(ValueError, match="empty"):
        image_settings(processor, model, "gemma4", tokenizer_config={**tokenizer_config, "eoi_token": ""}, snapshot=tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        image_settings(processor, model, "gemma4", tokenizer_config={**tokenizer_config, "eoi_token": "<|image>"}, snapshot=tmp_path)

    # Unknown token and multi-token string (Whitespace pre-tokenizer splits "<|image|> extra" into 2 tokens).
    with pytest.raises(ValueError, match="single vocabulary token"):
        image_settings(processor, model, "gemma4", tokenizer_config={**tokenizer_config, "image_token": "<missing>"}, snapshot=tmp_path)
    with pytest.raises(ValueError, match="single vocabulary token"):
        image_settings(processor, model, "gemma4", tokenizer_config={**tokenizer_config, "image_token": "<|image|> extra"}, snapshot=tmp_path)

    # Mismatched or bool config.json token IDs.
    with pytest.raises(ValueError, match="image_token_id"):
        image_settings(processor, {**model, "image_token_id": 999}, "gemma4", tokenizer_config=tokenizer_config, snapshot=tmp_path)
    with pytest.raises(ValueError, match="boi_token_id"):
        image_settings(processor, {**model, "boi_token_id": True}, "gemma4", tokenizer_config=tokenizer_config, snapshot=tmp_path)


def test_checked_in_gemma4_fixture_provenance_matches_image_settings(tmp_path):
    fixture_path = Path(__file__).resolve().parents[1] / "frontend" / "gemma4_images.json"
    fixture = json.loads(fixture_path.read_text())
    model = fixture["model"]
    vocab = {
        "<unk>": 0,
        "<|image|>": model["image_token_id"],
        "<|image>": model["boi_token_id"],
        "<image|>": model["eoi_token_id"],
    }
    tokenizer_config = {
        "image_token": "<|image|>",
        "boi_token": "<|image>",
        "eoi_token": "<image|>",
    }
    _write_synthetic_snapshot(
        tmp_path,
        processor=fixture["processor"],
        model=model,
        tokenizer_config=tokenizer_config,
        vocab=vocab,
    )
    resolved = image_settings(
        fixture["processor"],
        model,
        "gemma4",
        tokenizer_config=tokenizer_config,
        snapshot=tmp_path,
    )
    assert resolved == {
        "gemma4": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "image_token": "<|image|>",
            "start_token": "<|image>",
            "end_token": "<image|>",
        }
    }


@pytest.mark.parametrize(
    ("repository", "revision"),
    [
        ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
        ("nvidia/Gemma-4-31B-IT-NVFP4", "4135a98a9b728a548947683219633b25682223ac"),
        ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    ],
)
def test_cached_gemma4_checkpoints_prepare_image_settings(repository, revision):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / name).is_file():
            pytest.skip(f"{repository}@{revision} {name} is not cached under {hub}")
    prepared = prepare_frontend(
        snapshot,
        eos_token_ids=[1],
        image_profile="gemma4",
        preprocessor=snapshot / "processor_config.json",
    )
    assert prepared["image"] == {
        "gemma4": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "image_token": "<|image|>",
            "start_token": "<|image>",
            "end_token": "<image|>",
        }
    }
