/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Temporal RGB patch packing with configurable image envelope tokens.

use super::*;

/// Start/feature/end token envelope (`image_token`, `start_token`, `end_token`)
/// shared by `PatchMerge` and `Gemma4` image preparation profiles.
#[derive(Clone)]
pub(super) struct Envelope {
    image_token: String,
    start_token: String,
    end_token: String,
}

impl Envelope {
    pub(super) fn load(image_token: String, start_token: String, end_token: String) -> Result<Self> {
        if [&image_token, &start_token, &end_token]
            .iter()
            .any(|s| s.is_empty())
            || image_token == start_token
            || image_token == end_token
            || start_token == end_token
        {
            return Err("invalid image envelope tokens".into());
        }
        Ok(Self { image_token, start_token, end_token })
    }

    pub(super) fn expand(
        &self,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        let image_ids = frontend.encode(&self.image_token, false)?;
        let [image_id] = image_ids.as_slice() else {
            return Err("image placeholder must encode one token".into());
        };
        let envelope_id = |marker: &str| -> Result<u32> {
            let ids = frontend.encode(marker, false)?;
            match ids.as_slice() {
                [id] => Ok(*id),
                _ => Err("image envelope marker must encode one token".into()),
            }
        };
        let start_id = envelope_id(&self.start_token)?;
        let end_id = envelope_id(&self.end_token)?;
        let placeholder_count = prompt_token_ids
            .iter()
            .filter(|id| **id == *image_id)
            .count();
        if placeholder_count != images.geometry.len() {
            return Err("image placeholder count does not match input images".into());
        }
        let mut token_ids = Vec::new();
        let mut geometry = images.geometry.iter();
        for &id in prompt_token_ids {
            if id == *image_id {
                let tokens = geometry.next().expect("placeholder count was checked").tokens;
                token_ids.push(start_id);
                token_ids.extend(std::iter::repeat_n(*image_id, tokens));
                token_ids.push(end_id);
            } else {
                token_ids.push(id);
            }
        }
        let mm_token_type_ids = token_ids
            .iter()
            .map(|id| i32::from(*id == *image_id))
            .collect();
        Ok(ImagePrompt {
            position_ids: std::array::from_fn(|_| (0..token_ids.len() as i64).collect()),
            token_ids,
            mm_token_type_ids,
            rope_delta: 0,
        })
    }
}

#[derive(Clone)]
pub(super) struct Processor {
    pub(super) patches: patches::PatchProcessor,
    envelope: Envelope,
}

impl Processor {
    pub(super) fn load(settings: PatchMergeImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.patch_size != 14
            || settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.image_mean.iter().any(|x| !x.is_finite())
            || settings
                .image_std
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err("invalid temporal patch-merge image settings".into());
        }
        let envelope = Envelope::load(
            settings.image_token,
            settings.start_token,
            settings.end_token,
        )?;
        Ok(Self {
            patches: patches::PatchProcessor {
                limits,
                patch_size: settings.patch_size,
                min_pixels: settings.min_pixels,
                max_pixels: settings.max_pixels,
                normalization: patches::Normalization::ScaledChannel {
                    mean: settings.image_mean,
                    std: settings.image_std,
                },
                resize: resize::resize,
            },
            envelope,
        })
    }

    pub(super) fn patch_features(&self) -> usize {
        3 * 2 * self.patches.patch_size * self.patches.patch_size
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

    fn processor() -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::PatchMerge(PatchMergeImageSettings {
                patch_size: 14,
                min_pixels: 3136,
                max_pixels: 451584,
                image_mean: [0.48145466, 0.4578275, 0.40821073],
                image_std: [0.26862954, 0.26130258, 0.27577711],
                image_token: "image".into(),
                start_token: "start".into(),
                end_token: "end".into(),
            }),
            ImageLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn patch_geometry_and_duplicate_frames() {
        let processor = processor();
        assert_eq!(processor.geometry(672, 672).unwrap().patches, 2304);
        assert_eq!(processor.geometry(1, 1).unwrap().grid_thw, [1, 4, 4]);
        assert_eq!(processor.geometry(1008, 672).unwrap().grid_thw, [1, 38, 58]);
        let image =
            RgbImage::new(56, 56, (0..56 * 56 * 3).map(|x| (x % 256) as u8).collect()).unwrap();
        let images = processor.process(&[image]).unwrap();
        assert_eq!(images.pixel_values.len(), 16 * 1176);
        assert_eq!(&images.position_ids[..4], &[[0, 0], [0, 1], [1, 0], [1, 1]]);
        assert_eq!(images.cu_seqlens, [0, 16]);
        for patch in images.pixel_values.chunks_exact(1176) {
            for channel in patch.chunks_exact(392) {
                assert_eq!(&channel[..196], &channel[196..]);
            }
        }
        assert!((images.pixel_values[0] + 0.48145466 / 0.26862954).abs() < 1e-6);
    }
    #[test]
    fn native_pixels_match_canonical_torchvision_processor() {
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
        if !python.is_file() {
            return;
        }
        let script = r#"
import sys, struct, torch
from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor
image=(torch.arange(37*91*3,dtype=torch.int64)%256).to(torch.uint8).view(37,91,3).permute(2,0,1)
result=MiniMaxM3VLImageProcessor()(image,return_tensors='pt')
sys.stdout.buffer.write(struct.pack('<qqq',*result['image_grid_thw'][0].tolist()))
sys.stdout.buffer.write(result['pixel_values'].contiguous().numpy().tobytes())
"#;
        let reference = std::process::Command::new(python)
            .args(["-c", script])
            .output()
            .unwrap();
        assert!(
            reference.status.success(),
            "{}",
            String::from_utf8_lossy(&reference.stderr)
        );
        let image =
            RgbImage::new(91, 37, (0..37 * 91 * 3).map(|x| (x % 256) as u8).collect()).unwrap();
        let actual = processor().process(&[image]).unwrap();
        let grid: Vec<i64> = reference.stdout[..24]
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual.geometry[0].grid_thw.as_slice(), grid.as_slice());
        let pixels: Vec<f32> = reference.stdout[24..]
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual.pixel_values.len(), pixels.len());
        for (i, (a, b)) in actual.pixel_values.iter().zip(pixels).enumerate() {
            assert_eq!(*a, b, "pixel {i}");
        }
    }

    #[test]
    fn envelope_expands_image_placeholders_and_rejects_mismatches() {
        use crate::frontend::{Asset, FrontendAssets};
        use sha2::{Digest, Sha256};
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "flare-envelope-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = (0..128)
            .map(|id| ((id as u8 as char).to_string(), id))
            .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer
            .add_special_tokens(
                [
                    "<|image|>",
                    "<|image>",
                    "<image|>",
                    "]<]image[>[",
                    "]<]start of image[>[",
                    "]<]end of image[>[",
                ]
                .map(|text| tokenizers::AddedToken::from(text, true)),
            )
            .unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer.json", tokenizer.to_string(false).unwrap().as_bytes()),
                tokens: Default::default(),
                chat_template: Some(asset(
                    "chat_template.jinja",
                    b"{% mark %}{{ messages[0].content }}",
                )),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(&root).unwrap();

        for (image_tok, start_tok, end_tok) in [
            ("<|image|>", "<|image>", "<image|>"),
            ("]<]image[>[", "]<]start of image[>[", "]<]end of image[>["),
        ] {
            let envelope = Envelope::load(image_tok.into(), start_tok.into(), end_tok.into()).unwrap();
            let prompt = format!("A {image_tok} B {image_tok} C");
            let prompt_ids = frontend.encode(&prompt, false).unwrap();
            let processed = ProcessedImages {
                geometry: vec![
                    ImageGeometry { height: 48, width: 48, grid_thw: [1, 3, 3], patches: 9, tokens: 2 },
                    ImageGeometry { height: 48, width: 48, grid_thw: [1, 3, 3], patches: 9, tokens: 3 },
                ],
                ..Default::default()
            };
            let expanded = envelope.expand(&frontend, &prompt_ids, &processed).unwrap();
            let img_id = frontend.encode(image_tok, false).unwrap()[0];
            let start_id = frontend.encode(start_tok, false).unwrap()[0];
            let end_id = frontend.encode(end_tok, false).unwrap()[0];
            assert_eq!(expanded.token_ids.iter().filter(|id| **id == img_id).count(), 5);
            assert_eq!(expanded.token_ids.iter().filter(|id| **id == start_id).count(), 2);
            assert_eq!(expanded.token_ids.iter().filter(|id| **id == end_id).count(), 2);
            assert_eq!(expanded.mm_token_type_ids.iter().filter(|kind| **kind == 1).count(), 5);
            // Placeholder count mismatch fails cleanly.
            assert!(envelope.expand(&frontend, &prompt_ids, &ProcessedImages::default()).is_err());
        }

        // Duplicate or empty tokens fail load; multi-token marker fails expand.
        assert!(Envelope::load("".into(), "<|image>".into(), "<image|>".into()).is_err());
        assert!(Envelope::load("<|image|>".into(), "<|image|>".into(), "<image|>".into()).is_err());
        let bad = Envelope::load("ab".into(), "<|image>".into(), "<image|>".into()).unwrap();
        assert!(bad.expand(&frontend, &[], &ProcessedImages::default()).is_err());
    }
}
