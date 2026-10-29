//! Combining untrusted worker updates.
//!
//! Workers compile their own wasm, so any of them may send garbage. Clipping
//! and trimming bound how far a minority of bad updates can move the model;
//! they do not prove an update was computed honestly (that needs spot checks).

/// Robust mean of pseudo-gradients:
///
/// 1. scale each update down so its L2 norm is at most the median norm;
/// 2. take a coordinate-wise mean after dropping the `trim_fraction` largest
///    and smallest values at each coordinate.
pub fn robust_mean(updates: &[Vec<f32>], trim_fraction: f32) -> Vec<f32> {
    let n = updates.len();
    assert!(n > 0, "robust_mean needs at least one update");
    let dim = updates[0].len();

    let norms: Vec<f32> = updates.iter().map(|u| l2(u)).collect();
    let max_norm = median(&norms);
    let scales: Vec<f32> = norms
        .iter()
        .map(|&norm| {
            if norm > max_norm && norm > 0.0 {
                max_norm / norm
            } else {
                1.0
            }
        })
        .collect();

    let trim = ((n as f32 * trim_fraction).floor() as usize).min((n - 1) / 2);
    let kept = (n - 2 * trim) as f32;
    let mut column = vec![0.0f32; n];
    (0..dim)
        .map(|i| {
            for (c, (u, s)) in column.iter_mut().zip(updates.iter().zip(&scales)) {
                *c = u[i] * s;
            }
            if trim > 0 {
                column.sort_unstable_by(f32::total_cmp);
            }
            column[trim..n - trim].iter().sum::<f32>() / kept
        })
        .collect()
}

/// Outer optimiser from DiLoCo: SGD with Nesterov momentum on the pseudo-gradient.
pub struct Nesterov {
    pub lr: f32,
    pub momentum: f32,
    buf: Vec<f32>,
}

impl Nesterov {
    pub fn new(num_params: usize, lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            buf: vec![0.0; num_params],
        }
    }

    pub fn step(&mut self, params: &mut [f32], pseudo_grad: &[f32]) {
        for ((p, b), &g) in params.iter_mut().zip(&mut self.buf).zip(pseudo_grad) {
            *b = self.momentum * *b + g;
            *p -= self.lr * (g + self.momentum * *b);
        }
    }
}

pub fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_updates_average_normally() {
        let updates = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert_eq!(robust_mean(&updates, 0.2), vec![1.0, 2.0]);
    }

    #[test]
    fn one_huge_update_cannot_dominate() {
        let honest = vec![0.1, -0.1, 0.05];
        let mut updates = vec![honest.clone(); 4];
        updates.push(vec![1e6, 1e6, -1e6]);
        let mean = robust_mean(&updates, 0.2);
        for (m, h) in mean.iter().zip(&honest) {
            assert!((m - h).abs() < 1e-6, "{mean:?}");
        }
    }

    #[test]
    fn clipping_applies_without_trimming() {
        // With two workers nothing is trimmed, but the larger update is
        // scaled down to the median norm.
        let mean = robust_mean(&[vec![1.0], vec![100.0]], 0.2);
        assert!(mean[0] < 30.0, "{mean:?}");
    }

    #[test]
    fn nesterov_moves_against_the_pseudo_gradient() {
        let mut params = vec![1.0, 1.0];
        let mut opt = Nesterov::new(2, 0.5, 0.9);
        opt.step(&mut params, &[1.0, -1.0]);
        assert!(params[0] < 1.0 && params[1] > 1.0);
    }
}
