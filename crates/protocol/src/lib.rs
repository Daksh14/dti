//! Wire format shared by the worker (compiled to wasm) and the coordinator.
//!
//! JSON for control messages, little-endian binary for weights and updates.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ModelSpec {
    /// Byte-level bigram language model: a single 256 × 256 logit table.
    /// A placeholder until a transformer lands; it keeps the DiLoCo loop testable.
    ByteBigram,
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSpec::ByteBigram => f.write_str("byte-bigram"),
        }
    }
}

/// Hyperparameters for the local (inner) optimisation each worker runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub inner_steps: u32,
    pub batch_tokens: u32,
    pub inner_lr: f32,
    pub weight_decay: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub model: ModelSpec,
    pub num_params: u64,
    pub train: TrainConfig,
}

/// `POST /v1/join`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResponse {
    pub worker_id: String,
    pub run: RunInfo,
}

/// A byte range of the training corpus, fetched with `GET /v1/data/{offset}/{len}`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Shard {
    pub offset: u64,
    pub len: u64,
}

/// `GET /v1/task/{worker_id}`: what to train on for the current round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub round: u64,
    pub shard: Shard,
    /// Seeds batch sampling, so the coordinator can replay a worker's steps
    /// when spot-checking it.
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    BadMagic,
    Truncated,
    LengthMismatch { expected: usize, got: usize },
    NonFinite,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::BadMagic => f.write_str("bad magic"),
            CodecError::Truncated => f.write_str("truncated payload"),
            CodecError::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} values, got {got}")
            }
            CodecError::NonFinite => f.write_str("payload contains NaN or infinity"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Raw f32 weights, as served by `GET /v1/weights/{round}`.
pub fn encode_f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn decode_f32s(bytes: &[u8]) -> Result<Vec<f32>, CodecError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(CodecError::Truncated);
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(CodecError::NonFinite);
    }
    Ok(values)
}

const UPDATE_MAGIC: [u8; 4] = *b"DTQ1";
const UPDATE_BLOCK: usize = 256;
const UPDATE_HEADER: usize = 12;

/// Size in bytes of an encoded update for a model with `n` parameters.
pub fn update_len(n: usize) -> usize {
    UPDATE_HEADER + n.div_ceil(UPDATE_BLOCK) * 4 + n
}

/// Encodes a pseudo-gradient as int8 with one f32 scale per 256-value block,
/// about a quarter of the f32 size.
///
/// Layout: `"DTQ1" | n: u32 | block: u32 | scales: [f32; ceil(n/block)] | q: [i8; n]`.
pub fn encode_update(delta: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(update_len(delta.len()));
    out.extend_from_slice(&UPDATE_MAGIC);
    out.extend_from_slice(&(delta.len() as u32).to_le_bytes());
    out.extend_from_slice(&(UPDATE_BLOCK as u32).to_le_bytes());
    let scales: Vec<f32> = delta
        .chunks(UPDATE_BLOCK)
        .map(|block| block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0)
        .collect();
    for s in &scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for (block, &scale) in delta.chunks(UPDATE_BLOCK).zip(&scales) {
        for &v in block {
            let q = if scale > 0.0 {
                (v / scale).round().clamp(-127.0, 127.0)
            } else {
                0.0
            };
            out.push(q as i8 as u8);
        }
    }
    out
}

pub fn decode_update(bytes: &[u8], expected_len: usize) -> Result<Vec<f32>, CodecError> {
    if bytes.len() < UPDATE_HEADER {
        return Err(CodecError::Truncated);
    }
    if bytes[..4] != UPDATE_MAGIC {
        return Err(CodecError::BadMagic);
    }
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let block = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if n != expected_len {
        return Err(CodecError::LengthMismatch {
            expected: expected_len,
            got: n,
        });
    }
    if block != UPDATE_BLOCK || bytes.len() != update_len(n) {
        return Err(CodecError::Truncated);
    }
    let n_scales = n.div_ceil(UPDATE_BLOCK);
    let (scales, q) = bytes[UPDATE_HEADER..].split_at(n_scales * 4);
    let scales: Vec<f32> = scales
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if scales.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return Err(CodecError::NonFinite);
    }
    Ok(q.iter()
        .enumerate()
        .map(|(i, &b)| b as i8 as f32 * scales[i / UPDATE_BLOCK])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_roundtrip_is_within_quantization_error() {
        let delta: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) * 0.37).sin() * 0.01)
            .collect();
        let bytes = encode_update(&delta);
        assert_eq!(bytes.len(), update_len(delta.len()));
        let back = decode_update(&bytes, delta.len()).unwrap();
        for (block, back) in delta.chunks(UPDATE_BLOCK).zip(back.chunks(UPDATE_BLOCK)) {
            let max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (a, b) in block.iter().zip(back) {
                assert!((a - b).abs() <= max / 127.0 / 2.0 + 1e-9);
            }
        }
    }

    #[test]
    fn decode_update_rejects_bad_input() {
        let bytes = encode_update(&[1.0, -2.0, 3.0]);
        assert_eq!(
            decode_update(&bytes, 4),
            Err(CodecError::LengthMismatch {
                expected: 4,
                got: 3
            })
        );
        assert_eq!(
            decode_update(&bytes[..bytes.len() - 1], 3),
            Err(CodecError::Truncated)
        );

        let mut nan_scale = bytes.clone();
        nan_scale[UPDATE_HEADER..UPDATE_HEADER + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(decode_update(&nan_scale, 3), Err(CodecError::NonFinite));
    }

    #[test]
    fn f32_roundtrip() {
        let values = [0.0, -1.5, 3.25e-7];
        assert_eq!(decode_f32s(&encode_f32s(&values)).unwrap(), values);
        assert_eq!(decode_f32s(&[0, 0, 0]), Err(CodecError::Truncated));
    }
}
