//! Training steps through `wasi:nn`: the runner executes a TorchScript graph
//! (on its GPU, if the volunteer configured one) and the worker only moves
//! tensors in and out.
//!
//! The graph takes `params: f32[n]` and `tokens: u8[t]` and returns one f32
//! tensor `[loss, grad_0, ..., grad_{n-1}]`, because Wasmtime's PyTorch
//! backend returns a single output. `graphs/export.py` builds such graphs.

use crate::wasi::nn::errors::Error;
use crate::wasi::nn::graph;
use crate::wasi::nn::inference::GraphExecutionContext;
use crate::wasi::nn::tensor::{Tensor, TensorType};
use dti_protocol::{decode_f32s, encode_f32s};

pub struct Graph {
    ctx: GraphExecutionContext,
    num_params: usize,
}

impl Graph {
    /// Loads the graph the runner registered as `name`.
    pub fn load(name: &str, num_params: usize) -> Result<Self, String> {
        let err = |e: Error| format!("wasi:nn graph {name:?}: {}", describe(&e));
        let ctx = graph::load_by_name(name)
            .and_then(|g| g.init_execution_context())
            .map_err(err)?;
        Ok(Self { ctx, num_params })
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Mean next-token loss over `tokens`, writing its gradient into `grad`.
    pub fn loss_and_grad(
        &self,
        params: &[f32],
        tokens: &[u8],
        grad: &mut [f32],
    ) -> Result<f32, String> {
        let inputs = vec![
            (
                "params".to_string(),
                Tensor::new(
                    &[dim(params.len())?],
                    TensorType::Fp32,
                    &encode_f32s(params),
                ),
            ),
            (
                "tokens".to_string(),
                Tensor::new(&[dim(tokens.len())?], TensorType::U8, tokens),
            ),
        ];
        let outputs = self
            .ctx
            .compute(inputs)
            .map_err(|e| format!("wasi:nn compute: {}", describe(&e)))?;
        let [(_, output)] = <[_; 1]>::try_from(outputs)
            .map_err(|o| format!("wasi:nn compute: expected 1 output, got {}", o.len()))?;
        if output.ty() != TensorType::Fp32 {
            return Err(format!("wasi:nn output is {:?}, not FP32", output.ty()));
        }
        let values = decode_f32s(&output.data()).map_err(|e| format!("wasi:nn output: {e}"))?;
        let Some((&loss, g)) = values.split_first().filter(|(_, g)| g.len() == grad.len()) else {
            return Err(format!(
                "wasi:nn output has {} values, expected 1 + {}",
                values.len(),
                grad.len()
            ));
        };
        grad.copy_from_slice(g);
        Ok(loss)
    }
}

fn dim(len: usize) -> Result<u32, String> {
    u32::try_from(len).map_err(|_| format!("tensor of {len} values is too large for wasi:nn"))
}

fn describe(e: &Error) -> String {
    format!("{:?}: {}", e.code(), e.data())
}
