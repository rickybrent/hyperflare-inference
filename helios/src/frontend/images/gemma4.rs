/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Gemma 4 area-bounded resize, interleaved RGB patch packing, and envelopes.

use super::patch_merge::Envelope;
use super::*;

const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

#[derive(Clone)]
pub(super) struct Processor {
    limits: ImageLimits,
    patch_size: usize,
    pooling: usize,
    max_soft_tokens: usize,
    envelope: Envelope,
}

impl Processor {
    pub(super) fn load(settings: Gemma4ImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.patch_size == 0
            || settings.pooling_kernel_size == 0
            || settings.patch_size > 64
            || settings.pooling_kernel_size > 16
            || !SUPPORTED_SOFT_TOKENS.contains(&settings.max_soft_tokens)
        {
            return Err("invalid Gemma 4 image settings".into());
        }
        let envelope = Envelope::load(
            settings.image_token,
            settings.start_token,
            settings.end_token,
        )?;
        Ok(Self {
            limits,
            patch_size: settings.patch_size,
            pooling: settings.pooling_kernel_size,
            max_soft_tokens: settings.max_soft_tokens,
            envelope,
        })
    }

    fn side_multiple(&self) -> usize {
        self.patch_size * self.pooling
    }

    fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling * self.pooling
    }

    pub(super) fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        let patches_per_image = self.max_patches();
        let side = self.side_multiple();
        let target_pixels = (patches_per_image * self.patch_size * self.patch_size) as f64;
        let source_pixels = (width as f64) * (height as f64);
        let factor = (target_pixels / source_pixels).sqrt();
        let ideal_height = factor * height as f64;
        let ideal_width = factor * width as f64;
        let mut target_height = (ideal_height / side as f64).floor() as usize * side;
        let mut target_width = (ideal_width / side as f64).floor() as usize * side;
        if target_height == 0 && target_width == 0 {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image resizes below one Gemma 4 patch group",
            ))));
        }
        if target_height == 0 {
            target_height = side;
            target_width = (width / height).min(self.max_soft_tokens) * side;
        } else if target_width == 0 {
            target_width = side;
            target_height = (height / width).min(self.max_soft_tokens) * side;
        }
        if target_height * target_width > target_pixels as usize {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "Gemma 4 resize exceeds the soft-token budget",
            ))));
        }
        let grid_h = target_height / self.patch_size;
        let grid_w = target_width / self.patch_size;
        let patches = grid_h * grid_w;
        check_image_patches(self.limits, patches)?;
        Ok(ImageGeometry {
            height: target_height,
            width: target_width,
            grid_thw: [1, grid_h as i64, grid_w as i64],
            patches,
            tokens: patches / (self.pooling * self.pooling),
        })
    }

    pub(super) fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        let geometry = self.geometry(image.width, image.height)?;
        Ok((
            geometry,
            resize::resize(image, geometry.width, geometry.height),
        ))
    }

    pub(super) fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        self.limits.check_batch(images.len(), 0)?;
        let geometry: Vec<_> = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<_>>()?;
        let total: usize = geometry
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
            .ok_or_else(|| {
                Error::frontend(FrontendError::Input(Failure::new("patch count overflow")))
            })?;
        self.limits.check_batch(images.len(), total)?;
        let patch_features = 3 * self.patch_size * self.patch_size;
        let mut result = ProcessedImages {
            pixel_values: Vec::with_capacity(total.checked_mul(patch_features).ok_or(
                Error::frontend(FrontendError::Input(Failure::new(
                    "patch buffer extent overflow",
                ))),
            )?),
            geometry,
            position_ids: Vec::with_capacity(total),
            interpolation_indices: Vec::new(),
            interpolation_weights: Vec::new(),
            cu_seqlens: vec![0],
        };
        result.reserve_outputs(total, patch_features, false, images.len(), images.len() + 1)?;
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let pixels = resize::resize(image, geometry.width, geometry.height);
            let grid_h = geometry.height / self.patch_size;
            let grid_w = geometry.width / self.patch_size;
            // Gemma 4 patch vectors interleave RGB per pixel: (y, x, channel).
            for row in 0..grid_h {
                for column in 0..grid_w {
                    result.position_ids.push([column as i64, row as i64]);
                    for y in 0..self.patch_size {
                        for x in 0..self.patch_size {
                            let offset = ((row * self.patch_size + y) * geometry.width
                                + column * self.patch_size
                                + x)
                                * 3;
                            for channel in 0..3 {
                                result
                                    .pixel_values
                                    .push(pixels[offset + channel] as f32 * (1.0 / 255.0));
                            }
                        }
                    }
                }
            }
            result
                .cu_seqlens
                .push(result.cu_seqlens.last().unwrap() + geometry.patches as i32);
        }
        Ok(result)
    }

    pub(super) fn expand_prompt(
        &self,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        self.envelope.expand(frontend, prompt_token_ids, images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{Asset, FrontendAssets, images::ImageSettings};
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    fn processor() -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::Gemma4(Gemma4ImageSettings {
                patch_size: 16,
                pooling_kernel_size: 3,
                max_soft_tokens: 280,
                image_token: "<|image|>".into(),
                start_token: "<|image>".into(),
                end_token: "<image|>".into(),
            }),
            ImageLimits {
                max_patches_per_image: 2520,
                max_total_patches: 10080,
                patches_per_feature: 9,
                ..ImageLimits::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn geometry_matches_the_hugging_face_processor() {
        let processor = processor();
        for (width, height, grid, tokens) in [
            (1, 1, [1, 48, 48], 256),
            (672, 672, [1, 48, 48], 256),
            (1008, 672, [1, 39, 60], 260),
            (91, 37, [1, 30, 78], 260),
            (1920, 1080, [1, 36, 66], 264),
            (100, 3000, [1, 273, 9], 273),
            (3000, 100, [1, 9, 273], 273),
            (10, 3000, [1, 840, 3], 280),
            (3000, 10, [1, 3, 840], 280),
        ] {
            let geometry = processor.geometry(width, height).unwrap();
            assert_eq!(geometry.grid_thw, grid, "{width}x{height}");
            assert_eq!(geometry.tokens, tokens, "{width}x{height}");
            assert_eq!(geometry.tokens * 9, geometry.patches, "{width}x{height}");
        }
    }

    #[test]
    fn pixels_interleave_rgb_and_positions_are_xy() {
        let processor = processor();
        let pixels: Vec<u8> = (0..37 * 91 * 3).map(|x| (x % 256) as u8).collect();
        let image = RgbImage::new(91, 37, pixels).unwrap();
        let processed = processor.process(&[image]).unwrap();
        let geometry = processed.geometry[0];
        assert_eq!(geometry.grid_thw, [1, 30, 78]);
        assert_eq!(processed.pixel_values.len(), geometry.patches * 3 * 16 * 16);
        assert_eq!(processed.position_ids[0], [0, 0]);
        assert_eq!(processed.position_ids[1], [1, 0]);
        assert_eq!(
            processed.position_ids[geometry.grid_thw[2] as usize],
            [0, 1]
        );
        assert_eq!(processed.cu_seqlens, [0, geometry.patches as i32]);
    }

    #[test]
    fn checkpoint_tokens_expand_the_image_envelope() {
        let Some(snapshot) = std::env::var_os("FLARE_GEMMA4_TEST_SNAPSHOT").map(PathBuf::from)
        else {
            return;
        };
        if !snapshot.join("tokenizer.json").is_file() {
            return;
        }
        let asset = |name: &str| {
            let path = snapshot.join(name);
            Asset {
                sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
                path,
                range: None,
            }
        };
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
        if !python.is_file() {
            return;
        }
        let template =
            std::env::temp_dir().join(format!("gemma4-template-{}.jinja", std::process::id()));
        let patch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ignition/recipes/gemma4/assets/chat_template.google.patch.json");
        let script = r#"
import sys
from pathlib import Path
from ignition.preparation.frontend import downloaded_template_asset
Path(sys.argv[2]).write_text(downloaded_template_asset(Path(sys.argv[1]))["text"])
"#;
        let status = std::process::Command::new(&python)
            .current_dir(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap(),
            )
            .args([
                "-c",
                script,
                patch.to_str().unwrap(),
                template.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        if !status.success() {
            return;
        }
        let chat_template = Asset {
            sha256: format!("{:x}", Sha256::digest(std::fs::read(&template).unwrap())),
            path: template.clone(),
            range: None,
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer.json"),
                tokens: crate::frontend::test_support::tokens(&asset("tokenizer_config.json"))
                    .unwrap(),
                chat_template: Some(chat_template),
            },
            Default::default(),
        )
        .unwrap();
        let _ = std::fs::remove_file(&template);
        let processed = ProcessedImages {
            geometry: vec![ImageGeometry {
                height: 480,
                width: 1248,
                grid_thw: [1, 30, 78],
                patches: 2340,
                tokens: 260,
            }],
            cu_seqlens: vec![0, 2340],
            ..Default::default()
        };
        let prompt_ids = frontend.encode("Look <|image|> now", false).unwrap();
        let expanded = processor()
            .expand_prompt(&frontend, &prompt_ids, &processed)
            .unwrap();
        let boi = frontend.encode("<|image>", false).unwrap()[0];
        let image = frontend.encode("<|image|>", false).unwrap()[0];
        let eoi = frontend.encode("<image|>", false).unwrap()[0];
        let start = expanded.token_ids.iter().position(|id| *id == boi).unwrap();
        assert_eq!(expanded.token_ids[start], boi);
        assert_eq!(expanded.token_ids[start + 261], eoi);
        assert!(
            expanded.token_ids[start + 1..start + 261]
                .iter()
                .all(|id| *id == image)
        );
        assert!(
            expanded.mm_token_type_ids[start + 1..start + 261]
                .iter()
                .all(|kind| *kind == 1)
        );
        assert_eq!(expanded.mm_token_type_ids[start], 0);
    }

    #[test]
    fn native_pixels_and_positions_match_the_checkpoint_processor() {
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
        if !python.is_file() {
            return;
        }
        let script = r#"
import struct, sys, torch
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
w, h = int(sys.argv[1]), int(sys.argv[2])
image = (torch.arange(h * w * 3, dtype=torch.int64) % 256).to(torch.uint8).view(h, w, 3).permute(2, 0, 1)
result = Gemma4ImageProcessor()(images=[image], return_tensors="pt")
positions = result["image_position_ids"][0]
valid = (positions >= 0).all(-1)
patches = int(valid.sum())
grid_h = int(positions[..., 1].max()) + 1
grid_w = int(positions[..., 0].max()) + 1
sys.stdout.buffer.write(struct.pack('<qqqq', grid_h, grid_w, int(result["num_soft_tokens_per_image"][0]), patches))
sys.stdout.buffer.write(result["pixel_values"][0, :patches].contiguous().numpy().tobytes())
sys.stdout.buffer.write(positions[:patches].contiguous().numpy().tobytes())
"#;
        for (width, height) in [(91usize, 37usize), (10, 3000), (3000, 10)] {
            let reference = std::process::Command::new(&python)
                .args(["-c", script, &width.to_string(), &height.to_string()])
                .output()
                .unwrap();
            assert!(
                reference.status.success(),
                "{width}x{height}: {}",
                String::from_utf8_lossy(&reference.stderr)
            );
            let read = |offset: usize, count: usize| -> Vec<i64> {
                reference.stdout[offset..offset + count * 8]
                    .chunks_exact(8)
                    .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
                    .collect()
            };
            let header = read(0, 4);
            let (grid_h, grid_w, tokens, patches) = (
                header[0] as usize,
                header[1] as usize,
                header[2] as usize,
                header[3] as usize,
            );
            let image = RgbImage::new(
                width,
                height,
                (0..height * width * 3).map(|x| (x % 256) as u8).collect(),
            )
            .unwrap();
            let actual = processor().process(&[image]).unwrap();
            let geometry = actual.geometry[0];
            assert_eq!(geometry.grid_thw, [1, grid_h as i64, grid_w as i64], "{width}x{height}");
            assert_eq!(geometry.tokens, tokens, "{width}x{height}");
            assert_eq!(geometry.patches, patches, "{width}x{height}");
            assert_eq!(actual.cu_seqlens, [0, patches as i32], "{width}x{height}");
            let pixel_offset = 32;
            let pixels: Vec<f32> = reference.stdout[pixel_offset..pixel_offset + patches * 768 * 4]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            assert_eq!(actual.pixel_values.len(), pixels.len(), "{width}x{height}");
            for (i, (a, b)) in actual.pixel_values.iter().zip(&pixels).enumerate() {
                assert_eq!(*a, *b, "{width}x{height} pixel {i}");
            }
            let position_offset = pixel_offset + patches * 768 * 4;
            let positions: Vec<i64> = reference.stdout
                [position_offset..position_offset + patches * 2 * 8]
                .chunks_exact(8)
                .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
                .collect();
            assert_eq!(actual.position_ids.len(), patches, "{width}x{height}");
            for (i, (a, b)) in actual
                .position_ids
                .iter()
                .zip(positions.chunks_exact(2))
                .enumerate()
            {
                assert_eq!([a[0], a[1]], [b[0], b[1]], "{width}x{height} position {i}");
            }
        }
    }
}
