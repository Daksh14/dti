//! Models and the inner optimiser, in plain Rust so the same code runs in the
//! wasm worker and in the coordinator (which uses it to measure validation loss).

use dti_protocol::ModelSpec;

pub trait Model: Send + Sync {
    fn num_params(&self) -> usize;

    /// Initial weights for a new run.
    fn init(&self) -> Vec<f32>;

    /// Mean next-token loss over `tokens`, writing its gradient into `grad`.
    fn loss_and_grad(&self, params: &[f32], tokens: &[u8], grad: &mut [f32]) -> f32;

    fn loss(&self, params: &[f32], tokens: &[u8]) -> f32;
}

pub fn build(spec: ModelSpec) -> Box<dyn Model> {
    match spec {
        ModelSpec::ByteBigram => Box::new(ByteBigram),
    }
}

const VOCAB: usize = 256;

/// `params[a * 256 + b]` is the logit for byte `b` following byte `a`.
pub struct ByteBigram;

impl ByteBigram {
    /// Adds the loss for predicting `next` after `prev` and, if given, its gradient.
    fn accumulate(params: &[f32], prev: u8, next: u8, grad: Option<&mut [f32]>) -> f32 {
        let row = prev as usize * VOCAB;
        let logits = &params[row..row + VOCAB];
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
        let log_z = max + sum.ln();
        if let Some(grad) = grad {
            for (g, &v) in grad[row..row + VOCAB].iter_mut().zip(logits) {
                *g += (v - log_z).exp();
            }
            grad[row + next as usize] -= 1.0;
        }
        log_z - logits[next as usize]
    }
}

impl Model for ByteBigram {
    fn num_params(&self) -> usize {
        VOCAB * VOCAB
    }

    fn init(&self) -> Vec<f32> {
        vec![0.0; self.num_params()]
    }

    fn loss_and_grad(&self, params: &[f32], tokens: &[u8], grad: &mut [f32]) -> f32 {
        grad.fill(0.0);
        let pairs = tokens.len().saturating_sub(1);
        if pairs == 0 {
            return 0.0;
        }
        let mut loss = 0.0;
        for w in tokens.windows(2) {
            loss += Self::accumulate(params, w[0], w[1], Some(grad));
        }
        let scale = 1.0 / pairs as f32;
        grad.iter_mut().for_each(|g| *g *= scale);
        loss * scale
    }

    fn loss(&self, params: &[f32], tokens: &[u8]) -> f32 {
        let pairs = tokens.len().saturating_sub(1);
        if pairs == 0 {
            return 0.0;
        }
        let loss: f32 = tokens
            .windows(2)
            .map(|w| Self::accumulate(params, w[0], w[1], None))
            .sum();
        loss / pairs as f32
    }
}

/// AdamW. DiLoCo keeps this state on the worker across rounds.
pub struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(num_params: usize, weight_decay: f32) -> Self {
        Self {
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for i in 0..params.len() {
            let g = grad[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let update = (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + self.eps);
            params[i] -= lr * (update + self.weight_decay * params[i]);
        }
    }
}

/// SplitMix64: small, portable and identical on every platform, which matters
/// for replaying a worker's batches during verification.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`; `n` must be non-zero.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_gradient_matches_finite_differences() {
        let model = ByteBigram;
        let mut rng = Rng::new(7);
        let params: Vec<f32> = (0..model.num_params())
            .map(|_| (rng.below(1000) as f32 / 1000.0 - 0.5) * 0.1)
            .collect();
        let tokens = b"abracadabra";
        let mut grad = vec![0.0; params.len()];
        model.loss_and_grad(&params, tokens, &mut grad);

        let eps = 1e-2;
        for &(a, b) in &[(b'a', b'b'), (b'b', b'r'), (b'r', b'a'), (b'a', b'z')] {
            let i = a as usize * VOCAB + b as usize;
            let mut p = params.clone();
            p[i] += eps;
            let up = model.loss(&p, tokens);
            p[i] -= 2.0 * eps;
            let down = model.loss(&p, tokens);
            let numeric = (up - down) / (2.0 * eps);
            assert!((numeric - grad[i]).abs() < 1e-3, "{numeric} vs {}", grad[i]);
        }
    }

    #[test]
    fn adamw_reduces_loss() {
        let model = ByteBigram;
        let tokens = b"the quick brown fox jumps over the lazy dog. ".repeat(4);
        let mut params = model.init();
        let mut grad = vec![0.0; params.len()];
        let mut opt = AdamW::new(params.len(), 0.0);
        let before = model.loss(&params, &tokens);
        for _ in 0..50 {
            model.loss_and_grad(&params, &tokens, &mut grad);
            opt.step(&mut params, &grad, 0.1);
        }
        assert!(model.loss(&params, &tokens) < before * 0.5);
    }
}
