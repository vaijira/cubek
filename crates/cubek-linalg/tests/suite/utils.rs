use std::fmt::Display;

use cubecl::{TestRuntime, prelude::*};
use cubecl::std::tensor::{TensorHandle, into_contiguous};

pub(crate) fn tensorhandler_from_data<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    shape: Vec<usize>,
    data: &[F],
    dtype: cubecl::ir::Type,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(F::as_bytes(data));
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    TensorHandle::new(handle, shape, strides, dtype)
}

pub(crate) fn tensorhandler_from_data_col_major<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    shape: Vec<usize>,
    data: &[F],
    dtype: cubecl::ir::Type,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(F::as_bytes(data));
    let mut strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    TensorHandle::new(handle, shape, strides, dtype)
}

pub(crate) fn transpose_matrix<R: Runtime>(
    client: &ComputeClient<R>,
    matrix: &mut TensorHandle<R>,
) -> TensorHandle<R> {
    let mut strides = matrix.strides().to_vec();
    let mut shape = matrix.shape().to_vec();
    strides.swap(1, 0);
    shape.swap(1, 0);

    let transposed = TensorHandle::new(matrix.handle.clone(), shape, strides, matrix.dtype);

    into_contiguous::<R>(client, transposed.binding(), matrix.dtype)
}

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<F: Float + CubeElement + Display>(
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual_bytes = client.read_one(out.handle.clone()).unwrap();
    let actual = F::from_bytes(&actual_bytes);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        let allowed_error = (epsilon * e.to_f32().unwrap().abs()).max(epsilon);

        if f32::is_nan(a.to_f32().unwrap())
            || f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()) >= allowed_error
        {
            return Err(format!(
                "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                i,
                *a,
                *e,
                f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()),
                epsilon
            ));
        }
    }

    Ok(())
}
