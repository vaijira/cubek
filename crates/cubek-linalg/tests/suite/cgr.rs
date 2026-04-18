use std::fmt::Display;

use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle};


use crate::suite::utils::{assert_equals_approx, tensorhandler_from_data, transpose_matrix};

pub fn test_qr_cgr<F: Float + CubeElement + Display>(dim: u32) {
    let client = TestRuntime::client(&Default::default());
    let dim_usize = dim as usize;

    let shape = vec![dim as usize, dim as usize];
    let num_elements = shape.iter().product();
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }
    // let data = generate_deterministic_sparse_matrix::<F>(dim_usize, dim_usize, dim_usize - 1);

    let a = tensorhandler_from_data(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    /*let bytes = client.read_one_tensor(a.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("A Output => {output:?}"); */

    let (mut q_t, r) =
        match cubek_linalg::launch::<TestRuntime, F>(&cubek_linalg::QRStrategy::CommonGivensRotations, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(_) => (
                TensorHandle::empty(&client, shape.clone(), a.dtype),
                TensorHandle::empty(&client, shape.clone(), a.dtype),
            ),
        };

    let bytes = client.read_one(q_t.handle.clone()).unwrap();
    let output = F::from_bytes(&bytes);
    println!("Q Output => {output:?}");

    let bytes = client.read_one(r.handle.clone()).unwrap();
    let output = F::from_bytes(&bytes);
    println!("R Output => {output:?}");

    let q = transpose_matrix(&client, &mut q_t);

    let bytes_q = client.read_one(q.handle.clone()).unwrap();
    let q_vals = F::from_bytes(&bytes_q);

    let bytes_r = client.read_one(r.handle.clone()).unwrap();
    let r_vals = F::from_bytes(&bytes_r);

    let mut out_data = vec![F::from_int(0); num_elements];
    let dim_usize = dim as usize;
    for i in 0..dim_usize {
        for j in 0..dim_usize {
            let mut sum = F::from_int(0);
            for k in 0..dim_usize {
                sum += q_vals[i * dim_usize + k] * r_vals[k * dim_usize + j];
            }
            out_data[i * dim_usize + j] = sum;
        }
    }

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    println!("Result Output => {out_data:?}");

    if let Err(e) = assert_equals_approx(&client, &out, &data, 10e-3) {
        panic!("{}", e);
    }
}
pub fn test_qr_cgr_rect<F: Float + CubeElement + Display>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let shape = vec![rows_usize, cols_usize];
    let num_elements = rows_usize * cols_usize;
    let mut data = vec![F::from_int(1); num_elements];
    
    // Fill with some pattern
    for i in 0..rows_usize.min(cols_usize) {
        data[i * cols_usize + i] = F::from_int(2);
    }

    let a = tensorhandler_from_data(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    let (mut q_t, r) =
        match cubek_linalg::launch::<TestRuntime, F>(&cubek_linalg::QRStrategy::CommonGivensRotations, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(e) => panic!("QR launch failed: {:?}", e),
        };

    let q = transpose_matrix(&client, &mut q_t);

    let bytes_q = client.read_one(q.handle.clone()).unwrap();
    let q_vals = F::from_bytes(&bytes_q);
    let q_shape = q.shape();

    let bytes_r = client.read_one(r.handle.clone()).unwrap();
    let r_vals = F::from_bytes(&bytes_r);
    let r_shape = r.shape();

    let mut out_data = vec![F::from_int(0); num_elements];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            let mut sum = F::from_int(0);
            for k in 0..q_shape[1] {
                sum += q_vals[i * q_shape[1] + k] * r_vals[k * r_shape[1] + j];
            }
            out_data[i * cols_usize + j] = sum;
        }
    }

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &data, 10e-3) {
        panic!("{}", e);
    }
}
