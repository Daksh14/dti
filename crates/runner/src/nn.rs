//! `wasi:nn` for workers built with `--features nn`, backed by PyTorch.
//!
//! The worker can only `load-by-name` the graphs listed in the config, which
//! the runner reads and hash-checks itself. `load` from guest-supplied bytes is
//! refused (the context has no backends), because a TorchScript graph runs as
//! native code here, outside the sandbox: it could read files, for example.
//!
//! Input tensors are checked before PyTorch sees them. The backend copies
//! `product(dimensions)` elements out of a tensor's data without looking at its
//! length, so an unchecked guest tensor could make the runner read past the end
//! of a buffer.

use crate::config::{Graph, Target};
use crate::hex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use wasmtime::error::Context;
use wasmtime::{bail, format_err};
use wasmtime_wasi_nn::backend::pytorch::PytorchBackend;
use wasmtime_wasi_nn::backend::{
    BackendError, BackendExecutionContext, BackendGraph, BackendInner, Id, NamedTensor,
};
use wasmtime_wasi_nn::wit::{ExecutionTarget, TensorType, WasiNnCtx};
use wasmtime_wasi_nn::{ExecutionContext, GraphRegistry, Registry, Tensor};

/// Reads, checks and loads every configured graph.
pub fn load(graphs: &[Graph]) -> wasmtime::Result<WasiNnCtx> {
    let mut backend = PytorchBackend::default();
    let mut loaded = Graphs::default();
    for graph in graphs {
        let bytes = std::fs::read(&graph.path).with_context(|| {
            format!(
                "reading graph {:?} from {}",
                graph.name,
                graph.path.display()
            )
        })?;
        let sha256 = hex(&Sha256::digest(&bytes));
        if sha256 != graph.sha256 {
            bail!(
                "graph {:?} SHA-256 mismatch: expected {}, got {sha256}",
                graph.name,
                graph.sha256
            );
        }
        let target = match graph.target {
            Target::Cpu => ExecutionTarget::Cpu,
            // The backend would panic on the first input it moves to the GPU.
            Target::Gpu if !tch::Cuda::is_available() => bail!(
                "graph {:?} targets gpu, but libtorch finds no CUDA device \
                 (is it a CUDA build of libtorch?)",
                graph.name
            ),
            Target::Gpu => ExecutionTarget::Gpu,
        };
        let pytorch = backend
            .load(&[&bytes], target)
            .map_err(|e| format_err!("{e}: {}", source(&e)))
            .with_context(|| format!("loading graph {:?} with PyTorch", graph.name))?;
        let checked: Box<dyn BackendGraph> = Box::new(Checked(pytorch));
        loaded.0.insert(graph.name.clone(), checked.into());
    }
    Ok(WasiNnCtx::new(Vec::new(), Registry::from(loaded)))
}

#[derive(Default)]
struct Graphs(HashMap<String, wasmtime_wasi_nn::Graph>);

impl GraphRegistry for Graphs {
    fn get(&self, name: &str) -> Option<&wasmtime_wasi_nn::Graph> {
        self.0.get(name)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut wasmtime_wasi_nn::Graph> {
        self.0.get_mut(name)
    }
}

/// A graph whose execution contexts check every input tensor.
struct Checked<T>(T);

impl BackendGraph for Checked<wasmtime_wasi_nn::Graph> {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let ctx: Box<dyn BackendExecutionContext> =
            Box::new(Checked(self.0.init_execution_context()?));
        Ok(ctx.into())
    }
}

impl BackendExecutionContext for Checked<ExecutionContext> {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        check(tensor)?;
        self.0.set_input(id, tensor)
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        self.0.get_output(id)
    }

    fn compute(
        &mut self,
        inputs: Option<Vec<NamedTensor>>,
    ) -> Result<Option<Vec<NamedTensor>>, BackendError> {
        let result = inputs
            .iter()
            .flatten()
            .try_for_each(|input| check(&input.tensor))
            .and_then(|()| BackendExecutionContext::compute(&mut *self.0, inputs));
        // The guest only gets "Failed while accessing backend", so show why here.
        if let Err(e) = &result {
            println!("[nn]     compute failed: {}", source(e));
        }
        result
    }
}

/// Requires `data` to hold exactly the elements `dimensions` describe.
fn check(tensor: &Tensor) -> Result<(), BackendError> {
    let element = match tensor.ty {
        TensorType::U8 => 1,
        TensorType::Fp16 | TensorType::Bf16 => 2,
        TensorType::Fp32 | TensorType::I32 => 4,
        TensorType::Fp64 | TensorType::I64 => 8,
    };
    let expected = tensor
        .dimensions
        .iter()
        .try_fold(element, |n: usize, &d| n.checked_mul(d as usize));
    if expected != Some(tensor.data.len()) {
        return Err(BackendError::BackendAccess(format_err!(
            "{:?} tensor of shape {:?} needs {} bytes, got {}",
            tensor.ty,
            tensor.dimensions,
            expected.map_or_else(|| "too many".to_string(), |n| n.to_string()),
            tensor.data.len()
        )));
    }
    Ok(())
}

/// `BackendError`'s message is generic; the libtorch error is its source.
fn source(e: &dyn std::error::Error) -> String {
    e.source().map_or_else(String::new, |s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(dimensions: &[u32], ty: TensorType, len: usize) -> Tensor {
        Tensor::new(dimensions.to_vec(), ty, vec![0; len])
    }

    #[test]
    fn tensor_data_must_match_its_shape() {
        assert!(check(&tensor(&[2, 3], TensorType::Fp32, 24)).is_ok());
        assert!(check(&tensor(&[5], TensorType::U8, 5)).is_ok());
        assert!(check(&tensor(&[0], TensorType::I64, 0)).is_ok());
        assert!(check(&tensor(&[1 << 30], TensorType::Fp32, 4)).is_err());
        assert!(check(&tensor(&[2, 3], TensorType::Fp32, 25)).is_err());
        assert!(check(&tensor(&[u32::MAX; 4], TensorType::Fp64, 8)).is_err());
    }
}
