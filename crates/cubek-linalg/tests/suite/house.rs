use std::fmt::Display;

use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle};


use crate::suite::utils::{assert_equals_approx, tensorhandler_from_data, tensorhandler_from_data_col_major};

pub fn test_qr_baht<F: Float + CubeElement + Display>(dim: u32) {
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

    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &data,
        F::as_type_native_unchecked(),
    );

    let (q_t, r) =
        match cubek_linalg::launch::<TestRuntime, F>(&cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(_) => (
                TensorHandle::empty(&client, shape.clone(), a.dtype),
                TensorHandle::empty(&client, shape.clone(), a.dtype),
            ),
        };

    let bytes = client.read_one(q_t.handle.clone()).unwrap();
    let q_t_vals = F::from_bytes(&bytes);

    let bytes = client.read_one(r.handle.clone()).unwrap();
    let r_vals_out = F::from_bytes(&bytes);

    let mut out_data = vec![F::from_int(0); num_elements];
    let dim_usize = dim as usize;
    for i in 0..dim_usize {
        for j in 0..dim_usize {
            let mut sum = F::from_int(0);
            for k in 0..dim_usize {
                // q_t is Q in row-major. So Q_{i,k} is q_t_vals[i * dim + k]
                let q_ik = q_t_vals[i * dim_usize + k];
                // r is column-major. So R_{k,j} is r_vals[j * dim + k]
                let r_kj = r_vals_out[j * dim_usize + k];
                sum += q_ik * r_kj;
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
pub fn test_qr_baht_rect<F: Float + CubeElement + Display>(rows: u32, cols: u32) {
    let client = TestRuntime::client(&Default::default());
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let shape = vec![rows_usize, cols_usize];
    let num_elements = rows_usize * cols_usize;
    
    // Initialize data in row-major
    let mut row_major_data = vec![F::from_int(1); num_elements];
    for i in 0..rows_usize.min(cols_usize) {
        row_major_data[i * cols_usize + i] = F::from_int(2);
    }

    // Convert to column-major for BAHT input
    let mut col_major_data = vec![F::from_int(0); num_elements];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            col_major_data[j * rows_usize + i] = row_major_data[i * cols_usize + j];
        }
    }

    let a = tensorhandler_from_data_col_major(
        &client,
        shape.clone(),
        &col_major_data,
        F::as_type_native_unchecked(),
    );

    let (q_t, r) =
        match cubek_linalg::launch::<TestRuntime, F>(&cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(e) => panic!("QR launch failed: {:?}", e),
        };

    let bytes = client.read_one(q_t.handle.clone()).unwrap();
    let q_t_vals = F::from_bytes(&bytes);
    let q_t_shape = q_t.shape();

    let bytes = client.read_one(r.handle.clone()).unwrap();
    let r_vals_out = F::from_bytes(&bytes);
    let r_shape = r.shape();

    let mut out_data = vec![F::from_int(0); num_elements];
    for i in 0..rows_usize {
        for j in 0..cols_usize {
            let mut sum = F::from_int(0);
            for k in 0..q_t_shape[1] {
                // q_t is Q in row-major. So Q_{i,k} is q_t_vals[i * q_t_cols + k]
                let q_ik = q_t_vals[i * q_t_shape[1] + k];
                // r is column-major. So R_{k,j} is r_vals[j * r_rows + k]
                let r_kj = r_vals_out[j * r_shape[0] + k];
                sum += q_ik * r_kj;
            }
            out_data[i * cols_usize + j] = sum;
        }
    }

    let out = tensorhandler_from_data(&client, shape.clone(), &out_data, F::as_type_native_unchecked());

    if let Err(e) = assert_equals_approx(&client, &out, &row_major_data, 10e-3) {
        panic!("{}", e);
    }
}
