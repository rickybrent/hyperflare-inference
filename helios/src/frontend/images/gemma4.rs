/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Gemma 4 area-bounded resize, interleaved RGB patch packing, and envelopes.

use super::patch_merge::Envelope;
use super::*;

// Discrete budgets from Hugging Face's `image_processor.max_soft_tokens`
// (validated here as wire-level defense-in-depth alongside Ignition's checkpoint gate).
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
        // Defensive wire caps prevent pathological patch/pooling allocations from malformed payloads.
        if settings.patch_size == 0 || settings.patch_size > 64 {
            return Err(format!(
                "Gemma 4 patch_size ({}) must be in 1..=64",
                settings.patch_size
            )
            .into());
        }
        if settings.pooling_kernel_size == 0 || settings.pooling_kernel_size > 16 {
            return Err(format!(
                "Gemma 4 pooling_kernel_size ({}) must be in 1..=16",
                settings.pooling_kernel_size
            )
            .into());
        }
        if !SUPPORTED_SOFT_TOKENS.contains(&settings.max_soft_tokens) {
            return Err(format!(
                "unsupported Gemma 4 max_soft_tokens ({}); expected one of {:?}",
                settings.max_soft_tokens, SUPPORTED_SOFT_TOKENS
            )
            .into());
        }
        let pooling_area = settings
            .pooling_kernel_size
            .checked_mul(settings.pooling_kernel_size)
            .ok_or("Gemma 4 pooling area overflow")?;
        if pooling_area != limits.patches_per_feature {
            return Err(format!(
                "Gemma 4 pooling area ({}x{} = {pooling_area}) differs from prepared patches_per_feature ({})",
                settings.pooling_kernel_size,
                settings.pooling_kernel_size,
                limits.patches_per_feature
            )
            .into());
        }
        let max_patches = settings
            .max_soft_tokens
            .checked_mul(pooling_area)
            .ok_or("Gemma 4 max patch budget overflow")?;
        if max_patches > limits.max_patches_per_image {
            return Err(format!(
                "Gemma 4 max patch budget ({} * {pooling_area} = {max_patches}) exceeds prepared max_patches_per_image ({})",
                settings.max_soft_tokens, limits.max_patches_per_image
            )
            .into());
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

    pub(super) fn patch_features(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    pub(super) fn max_envelope_tokens(&self) -> usize {
        self.max_soft_tokens + 2
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
        let patch_features = self.patch_features();
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

    fn processor_with_budget(max_soft_tokens: usize) -> ImageProcessor {
        let max_patches = max_soft_tokens * 9;
        ImageProcessor::load(
            ImageSettings::Gemma4(Gemma4ImageSettings {
                patch_size: 16,
                pooling_kernel_size: 3,
                max_soft_tokens,
                image_token: "<|image|>".into(),
                start_token: "<|image>".into(),
                end_token: "<image|>".into(),
            }),
            ImageLimits {
                max_patches_per_image: max_patches,
                max_total_patches: max_patches * 4,
                patches_per_feature: 9,
                ..ImageLimits::default()
            },
        )
        .unwrap()
    }

    fn processor() -> ImageProcessor {
        processor_with_budget(280)
    }

    #[test]
    fn load_validates_pooling_and_patch_budget_limits() {
        let settings = Gemma4ImageSettings {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
            image_token: "<|image|>".into(),
            start_token: "<|image>".into(),
            end_token: "<image|>".into(),
        };
        let pooling_err = ImageProcessor::load(
            ImageSettings::Gemma4(settings.clone()),
            ImageLimits {
                patches_per_feature: 4,
                max_patches_per_image: 2520,
                ..ImageLimits::default()
            },
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            pooling_err.contains("patches_per_feature")
                && pooling_err.contains("9")
                && pooling_err.contains("4"),
            "{pooling_err}"
        );
        let budget_err = ImageProcessor::load(
            ImageSettings::Gemma4(settings.clone()),
            ImageLimits {
                patches_per_feature: 9,
                max_patches_per_image: 2519,
                ..ImageLimits::default()
            },
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            budget_err.contains("max_patches_per_image")
                && budget_err.contains("2520")
                && budget_err.contains("2519"),
            "{budget_err}"
        );
        let soft_err = ImageProcessor::load(
            ImageSettings::Gemma4(Gemma4ImageSettings {
                max_soft_tokens: 200,
                ..settings
            }),
            ImageLimits {
                patches_per_feature: 9,
                max_patches_per_image: 2520,
                ..ImageLimits::default()
            },
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            soft_err.contains("max_soft_tokens") && soft_err.contains("200"),
            "{soft_err}"
        );
    }

    #[test]
    fn canonical_processor_pixels_positions_and_resize() {
        use serde_json::{Value, json};
        let fixture: Value =
            serde_json::from_str(include_str!("../../../../tests/frontend/gemma4_images.json"))
                .unwrap();
        let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        for case in fixture["cases"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as usize;
            let height = case["height"].as_u64().unwrap() as usize;
            let seed = case["seed"].as_u64().unwrap() as usize;
            let max_soft_tokens = case["max_soft_tokens"].as_u64().unwrap() as usize;
            let processor = processor_with_budget(max_soft_tokens);
            let pixels: Vec<u8> = (0..height)
                .flat_map(|y| {
                    (0..width).flat_map(move |x| {
                        [(x * 7 + seed * 31) as u8, (y * 11) as u8, (x + y * 3) as u8]
                    })
                })
                .collect();
            let image = RgbImage::new(width, height, pixels).unwrap();
            let (geometry, resized) = processor.resized_rgb(&image).unwrap();
            assert_eq!(
                json!(geometry.grid_thw),
                case["grid"],
                "{width}x{height}@{max_soft_tokens}"
            );
            assert_eq!(
                geometry.tokens as u64,
                case["tokens"].as_u64().unwrap(),
                "{width}x{height}@{max_soft_tokens}"
            );
            assert_eq!(
                geometry.patches as u64,
                case["patches"].as_u64().unwrap(),
                "{width}x{height}@{max_soft_tokens}"
            );
            assert_eq!(
                hash(&resized),
                case["resized"],
                "resize {width}x{height}@{max_soft_tokens}"
            );
            let processed = processor.process(&[image]).unwrap();
            let pixels: Vec<u8> = processed
                .pixel_values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                hash(&pixels),
                case["pixels"],
                "pixels {width}x{height}@{max_soft_tokens}"
            );
            let positions: Vec<u8> = processed
                .position_ids
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                hash(&positions),
                case["positions"],
                "positions {width}x{height}@{max_soft_tokens}"
            );
            assert_eq!(processed.cu_seqlens, [0, geometry.patches as i32]);
            assert!(processed.interpolation_indices.is_empty());
        }
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
    #[ignore = "requires FLARE_GEMMA4_TEST_SNAPSHOT"]
    fn checkpoint_tokens_expand_the_image_envelope() {
        let snapshot = PathBuf::from(
            std::env::var_os("FLARE_GEMMA4_TEST_SNAPSHOT")
                .expect("FLARE_GEMMA4_TEST_SNAPSHOT must be set"),
        );
        assert!(
            snapshot.join("tokenizer.json").is_file(),
            "missing tokenizer.json under {}",
            snapshot.display()
        );
        let asset = |name: &str| {
            let path = snapshot.join(name);
            Asset {
                sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
                path,
                range: None,
            }
        };
        // Frontend::load requires a chat template asset for PromptEncoder::Jinja (frontend/mod.rs).
        struct TempFile(PathBuf);
        impl Drop for TempFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let template = TempFile(
            std::env::temp_dir().join(format!("gemma4-template-{}.jinja", std::process::id())),
        );
        let template_bytes = b"{% mark %}{{ messages[0].content }}";
        std::fs::write(&template.0, template_bytes).unwrap();
        let chat_template = Asset {
            sha256: format!("{:x}", Sha256::digest(template_bytes)),
            path: template.0.clone(),
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
        drop(template);
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
        assert_eq!(expanded.mm_token_type_ids[start + 261], 0);
    }
}
