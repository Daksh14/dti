//! The training data, cut into slices, and which worker is training on which.
//!
//! Workers download slices themselves (from the coordinator's corpus, or as
//! parquet row groups from Hugging Face), one at a time and no more than their
//! volunteer's data budget allows. The coordinator decides what is trained on:
//! the slices, their order, and, through the dataset plan's filters, which
//! documents in them.

use dti_protocol::{Slice, Source};
use serde::Deserialize;
use std::path::Path;

/// `data/plan.py`'s output.
#[derive(Deserialize)]
pub struct Plan {
    /// Where the data comes from, for display.
    pub source: String,
    /// Held-out text for validation loss, relative to the plan file.
    pub val: String,
    pub slices: Vec<PlannedSlice>,
}

#[derive(Deserialize)]
pub struct PlannedSlice {
    pub bytes: u64,
    pub source: Source,
}

impl Plan {
    /// Reads a plan and its validation text.
    pub fn load(path: &Path) -> anyhow::Result<(Self, Vec<u8>)> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading dataset plan {}: {e}", path.display()))?;
        let plan: Plan = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing dataset plan {}: {e}", path.display()))?;
        let val_path = path.parent().unwrap_or(Path::new("")).join(&plan.val);
        let val = std::fs::read(&val_path)
            .map_err(|e| anyhow::anyhow!("reading validation text {}: {e}", val_path.display()))?;
        anyhow::ensure!(
            val.len() >= 2,
            "validation text {} is empty",
            val_path.display()
        );
        anyhow::ensure!(!plan.slices.is_empty(), "dataset plan has no slices");
        Ok((plan, val))
    }
}

/// Cuts `len` bytes of the coordinator's corpus into slices of `slice_bytes`.
pub fn corpus_slices(len: u64, slice_bytes: u64) -> Vec<PlannedSlice> {
    (0..len)
        .step_by(slice_bytes as usize)
        .map(|offset| {
            let len = slice_bytes.min(len - offset);
            PlannedSlice {
                bytes: len,
                source: Source::Corpus { offset, len },
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Free,
    Leased(String),
    Done,
}

pub struct Ledger {
    slices: Vec<Slice>,
    state: Vec<State>,
}

impl Ledger {
    /// Repeats the slices `epochs` times and shuffles them, so each worker
    /// sees a mix of the whole dataset rather than one file after another.
    pub fn new(slices: Vec<PlannedSlice>, epochs: u32) -> Self {
        let mut all: Vec<&PlannedSlice> = (0..epochs).flat_map(|_| &slices).collect();
        let mut rng = dti_model::Rng::new(0xDA7A);
        for i in (1..all.len()).rev() {
            all.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let slices: Vec<Slice> = all
            .into_iter()
            .enumerate()
            .map(|(id, s)| Slice {
                id: id as u64,
                bytes: s.bytes,
                source: s.source.clone(),
            })
            .collect();
        Self {
            state: vec![State::Free; slices.len()],
            slices,
        }
    }

    /// Leases `worker` the next free slice of at most `max_bytes`.
    pub fn lease(&mut self, worker: &str, max_bytes: u64) -> Option<Slice> {
        let i = (0..self.slices.len())
            .find(|&i| self.state[i] == State::Free && self.slices[i].bytes <= max_bytes)?;
        self.state[i] = State::Leased(worker.to_string());
        Some(self.slices[i].clone())
    }

    /// Records that `worker` has trained on all of slice `id` and its update
    /// was accepted. Returns the slice's size, or `None` if it was not leased
    /// to `worker` (it may have been released and leased again meanwhile).
    pub fn done(&mut self, worker: &str, id: u64) -> Option<u64> {
        let i = usize::try_from(id)
            .ok()
            .filter(|&i| i < self.slices.len())?;
        match &self.state[i] {
            State::Leased(w) if w == worker => {
                self.state[i] = State::Done;
                Some(self.slices[i].bytes)
            }
            _ => None,
        }
    }

    /// Puts the slices leased to `worker` back in the pool. Returns how many.
    pub fn release(&mut self, worker: &str) -> usize {
        let mut released = 0;
        for state in &mut self.state {
            if matches!(state, State::Leased(w) if w == worker) {
                *state = State::Free;
                released += 1;
            }
        }
        released
    }

    pub fn len(&self) -> usize {
        self.slices.len()
    }

    pub fn done_count(&self) -> usize {
        self.state.iter().filter(|s| **s == State::Done).count()
    }

    pub fn leased_count(&self) -> usize {
        self.state
            .iter()
            .filter(|s| matches!(s, State::Leased(_)))
            .count()
    }

    pub fn finished(&self) -> bool {
        self.state.iter().all(|s| *s == State::Done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger(sizes: &[u64]) -> Ledger {
        let slices = sizes
            .iter()
            .map(|&bytes| PlannedSlice {
                bytes,
                source: Source::Corpus {
                    offset: 0,
                    len: bytes,
                },
            })
            .collect();
        Ledger::new(slices, 1)
    }

    #[test]
    fn leases_respect_the_budget_and_each_slice_goes_to_one_worker() {
        let mut l = ledger(&[10, 10, 50]);
        assert!(l.lease("w1", 5).is_none());
        let a = l.lease("w1", 10).unwrap();
        let b = l.lease("w2", 10).unwrap();
        assert_ne!(a.id, b.id);
        assert!(
            l.lease("w2", 10).is_none(),
            "only the 50-byte slice is left"
        );
        assert_eq!(l.lease("w2", 100).unwrap().bytes, 50);
        assert!(l.lease("w3", u64::MAX).is_none());
    }

    #[test]
    fn only_the_leaseholder_can_finish_a_slice() {
        let mut l = ledger(&[10, 10]);
        let a = l.lease("w1", 10).unwrap();
        assert_eq!(l.done("w2", a.id), None);
        assert_eq!(l.done("w1", a.id), Some(10));
        assert_eq!(l.done("w1", a.id), None, "already done");
        assert_eq!(l.done("w1", 99), None);
        assert!(!l.finished());
        let b = l.lease("w1", 10).unwrap();
        l.done("w1", b.id);
        assert!(l.finished());
    }

    #[test]
    fn released_slices_go_back_to_the_pool() {
        let mut l = ledger(&[10, 10]);
        let a = l.lease("w1", 10).unwrap();
        l.lease("w1", 10).unwrap();
        assert_eq!(l.release("w1"), 2);
        assert_eq!(l.done("w1", a.id), None);
        assert_eq!(l.leased_count(), 0);
        assert!(l.lease("w2", 10).is_some());
    }

    #[test]
    fn corpus_slices_cover_the_corpus() {
        let s = corpus_slices(25, 10);
        let lens: Vec<u64> = s.iter().map(|s| s.bytes).collect();
        assert_eq!(lens, [10, 10, 5]);
        assert!(matches!(s[2].source, Source::Corpus { offset: 20, len: 5 }));
    }

    #[test]
    fn epochs_repeat_the_data() {
        let slices = corpus_slices(30, 10);
        assert_eq!(Ledger::new(slices, 3).len(), 9);
    }
}
