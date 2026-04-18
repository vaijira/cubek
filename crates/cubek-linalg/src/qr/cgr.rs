use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

// Followed algorithm described in page 2 and 3 in
// https://thesai.org/Downloads/Volume11No5/Paper_78-Parallel_QR_Factorization_using_Givens_Rotations.pdf

// Fill vector l with the col_index from matrix r.
#[cube(launch, launch_unchecked)]
fn get_column_from_matrix<F: Float>(col_index: u32, r: &Tensor<F>, l: &mut Tensor<F>) {
    let n_cols = r.shape(1);
    if ABSOLUTE_POS < l.len() {
        l[ABSOLUTE_POS] = r[ABSOLUTE_POS * n_cols + col_index as usize]
    }
}

#[cube]
fn givens_rotation<F: Float>(a: F, b: F) -> (F, F) {
    let zero = F::from_int(0);
    let one = F::from_int(1);

    if b == zero {
        (one, zero)
    } else if a == zero {
        (zero, one)
    } else {
        let abs_a = F::abs(a);
        let abs_b = F::abs(b);
        if abs_a > abs_b {
            let r = b / a;
            let c = one / F::sqrt(fma(r, r, one));
            let s = c * r;
            (c, s)
        } else {
            let r = a / b;
            let s = one / F::sqrt(fma(r, r, one));
            let c = s * r;
            (c, s)
        }
    }
}

#[cube(launch, launch_unchecked)]
fn qr_column_parallel<F: Float>(
    col_index: u32,
    l: &Tensor<F>,
    q: &mut Tensor<F>,
    r: &mut Tensor<F>,
) {
    let n_rows = r.shape(0);
    let n_cols = r.shape(1);
    let q_rows = q.shape(0);

    if ABSOLUTE_POS < q_rows {
        let mut pivot_r = l[col_index as usize];

        for k in (col_index + 1) as usize..n_rows {
            let b = l[k];
            let (c, s) = givens_rotation::<F>(pivot_r, b);
            pivot_r = c * pivot_r + s * b;

            // Update R
            if ABSOLUTE_POS < n_cols && ABSOLUTE_POS >= col_index as usize {
                let row_col_val = r[col_index as usize * n_cols + ABSOLUTE_POS];
                let row_k_val = r[k * n_cols + ABSOLUTE_POS];
                r[col_index as usize * n_cols + ABSOLUTE_POS] = c * row_col_val + s * row_k_val;
                r[k * n_cols + ABSOLUTE_POS] = -s * row_col_val + c * row_k_val;
            }

            // Update Q^T
            let q_row_col_val = q[col_index as usize * q_rows + ABSOLUTE_POS];
            let q_row_k_val = q[k * q_rows + ABSOLUTE_POS];
            q[col_index as usize * q_rows + ABSOLUTE_POS] = c * q_row_col_val + s * q_row_k_val;
            q[k * q_rows + ABSOLUTE_POS] = -s * q_row_col_val + c * q_row_k_val;
        }
    }
}

/// Launch QR decomposition common Givens rotation kernels.
pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q: &TensorHandle<R>,
    r: &TensorHandle<R>,
) {
    let n_rows = r.shape()[0];
    let n_cols = r.shape()[1];
    let l = TensorHandle::<R>::zeros(client, vec![n_rows], r.dtype);
    let max_cube_dim = client.properties().hardware.max_cube_dim.0;

    for col in 0..(n_cols.min(n_rows - 1)) {
        let col_u32 = col as u32;

        // 1. Copy reference column to l
        let cube_dim_l = CubeDim::new_1d((n_rows as u32).min(max_cube_dim));
        let cube_count_l = calculate_cube_count_elemwise(client, n_rows, cube_dim_l);
        unsafe {
            get_column_from_matrix::launch_unchecked::<E, R>(
                client,
                cube_count_l,
                cube_dim_l,
                col_u32,
                r.clone().into_arg(),
                l.clone().into_arg(),
            );
        }

        // 2. Perform parallel Givens rotation pass
        let cube_dim = CubeDim::new_1d((n_rows as u32).min(max_cube_dim));
        let cube_count = calculate_cube_count_elemwise(client, n_rows, cube_dim);

        unsafe {
            qr_column_parallel::launch_unchecked::<E, R>(
                client,
                cube_count,
                cube_dim,
                col_u32,
                l.clone().into_arg(),
                q.clone().into_arg(),
                r.clone().into_arg(),
            );
        }
    }
}
