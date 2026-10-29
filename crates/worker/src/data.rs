//! This worker's share of the training data.
//!
//! The coordinator leases slices one at a time, each no larger than what is
//! left of the volunteer's data budget, and the worker downloads them as
//! training needs more text: from the coordinator's corpus, or one parquet row
//! group's text column from Hugging Face. Every byte is trained on once, in
//! order, so the volunteer's budget sets how much compute they give.

use crate::http::Client;
use crate::parquet;
use dti_protocol::{Slice, SliceRequest, Source};
use std::collections::VecDeque;

pub struct Data<'a> {
    client: &'a Client,
    worker_id: &'a str,
    /// Download bytes left in the volunteer's budget.
    budget: u64,
    /// Leased but not downloaded yet, e.g. because the download failed.
    leased: Option<Slice>,
    /// Text since the last commit; training has read up to `pos`.
    text: Vec<u8>,
    pos: usize,
    /// Slices whose text is in `text`, with where their text ends.
    loaded: VecDeque<(u64, usize)>,
    /// The budget is spent or the coordinator has nothing left that fits.
    exhausted: bool,
}

impl<'a> Data<'a> {
    pub fn new(client: &'a Client, worker_id: &'a str, budget: u64) -> Self {
        Self {
            client,
            worker_id,
            budget,
            leased: None,
            text: Vec::new(),
            pos: 0,
            loaded: VecDeque::new(),
            exhausted: false,
        }
    }

    pub fn budget_left(&self) -> u64 {
        self.budget
    }

    /// The next `n` bytes of text (fewer at the very end), or `None` once the
    /// budget or the dataset is used up.
    pub fn next_batch(&mut self, n: usize) -> Result<Option<&[u8]>, String> {
        while self.text.len() - self.pos < n && !self.exhausted {
            self.load_next()?;
        }
        let start = self.pos;
        let end = (start + n).min(self.text.len());
        if end - start < 2 {
            // Too little left for a next-token pair; count it as read.
            self.pos = self.text.len();
            return Ok(None);
        }
        self.pos = end;
        Ok(Some(&self.text[start..end]))
    }

    /// The text read so far is trained on and its round's update accepted.
    /// Returns the slices that are now completely used.
    pub fn commit(&mut self) -> Vec<u64> {
        let mut done = Vec::new();
        while let Some(&(id, end)) = self.loaded.front() {
            if end > self.pos {
                break;
            }
            done.push(id);
            self.loaded.pop_front();
        }
        self.text.drain(..self.pos);
        for (_, end) in &mut self.loaded {
            *end -= self.pos;
        }
        self.pos = 0;
        done
    }

    /// Goes back to the last commit, so a failed round's text is used again.
    pub fn rewind(&mut self) {
        self.pos = 0;
    }

    fn load_next(&mut self) -> Result<(), String> {
        let slice = match self.leased.take() {
            Some(slice) => slice,
            None => match self.lease()? {
                Some(slice) => slice,
                None => {
                    self.exhausted = true;
                    return Ok(());
                }
            },
        };
        let text = match self.download(&slice) {
            Ok(text) => text,
            Err(e) => {
                self.leased = Some(slice);
                return Err(e);
            }
        };
        self.budget -= slice.bytes;
        self.text.extend_from_slice(&text);
        self.loaded.push_back((slice.id, self.text.len()));
        Ok(())
    }

    fn lease(&self) -> Result<Option<Slice>, String> {
        let req = serde_json::to_vec(&SliceRequest {
            max_bytes: self.budget,
        })
        .map_err(|e| e.to_string())?;
        let r = self.client.post(
            &format!("/v1/slice/{}", self.worker_id),
            "application/json",
            &req,
        )?;
        match r.status {
            204 => Ok(None),
            200 => {
                let slice: Slice =
                    serde_json::from_slice(&r.body).map_err(|e| format!("slice: {e}"))?;
                if slice.bytes > self.budget {
                    return Err(format!(
                        "coordinator leased slice {} of {} bytes, over the {} left in the budget",
                        slice.id, slice.bytes, self.budget
                    ));
                }
                Ok(Some(slice))
            }
            status => Err(format!(
                "slice: HTTP {status} {}",
                String::from_utf8_lossy(&r.body).trim()
            )),
        }
    }

    fn download(&self, slice: &Slice) -> Result<Vec<u8>, String> {
        match &slice.source {
            Source::Corpus { offset, len } => {
                let r = self.client.get(&format!("/v1/data/{offset}/{len}"))?;
                if r.status != 200 || r.body.len() as u64 != *len {
                    return Err(format!("slice {}: HTTP {}", slice.id, r.status));
                }
                Ok(r.body)
            }
            Source::Parquet(chunk) => {
                let raw = self.client.get_range(&chunk.url, chunk.offset, chunk.len)?;
                parquet::text(chunk, raw)
            }
        }
    }
}
