use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::qr::{QRSetupError, QRStrategy};


/// Kernel to compute y = Q_already_t * b
/// Q_already_t is m x m row-major. b is m x 1.
/// y_i = sum_j (Q_already_t)_ij * b_j
/// (Q_already_t)_ij is at index i * m + j.
#[cube(launch_unchecked)]
pub fn q_already_t_b_kernel<F: Float>(m: u32, q_t: &Tensor<F>, b: &Tensor<F>, y: &mut Tensor<F>) {
    let i = ABSOLUTE_POS_X;
    if i < m {
        let mut sum = 0.0f64;
        for j in 0..m {
            sum = fma(
                f64::cast_from(q_t[(j * m + i) as usize]),
                f64::cast_from(b[j as usize]),
                sum,
            );
        }
        y[i as usize] = F::cast_from(sum);
    }
}

/// Back substitution for Rx = y where R is upper triangular.
/// Accumulates in f64 to minimise rounding in the triangular solve.
#[cube(launch_unchecked)]
pub fn back_substitution_kernel<F: Float>(
    n: u32,
    rows: u32,
    r: &Tensor<F>,
    y: &Tensor<F>,
    x: &mut Tensor<F>,
    is_col_major: u32,
) {
    if ABSOLUTE_POS == 0 {
        let mut i = n;
        while i > 0 {
            i -= 1;
            let mut sum = 0.0f64;
            for j in (i + 1)..n {
                let r_idx = if is_col_major == 1 {
                    j * rows + i
                } else {
                    i * n + j
                };
                sum = fma(
                    f64::cast_from(r[r_idx as usize]),
                    f64::cast_from(x[j as usize]),
                    sum,
                );
            }
            let diag_idx = if is_col_major == 1 {
                i * rows + i
            } else {
                i * n + i
            };
            x[i as usize] = F::cast_from(
                (f64::cast_from(y[i as usize]) - sum)
                    / f64::cast_from(r[diag_idx as usize]),
            );
        }
    }
}

/// Solve Ax = b using QR decomposition.
pub fn solve<R: Runtime, E: Float + CubeElement>(
    strategy: &QRStrategy,
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
    b: &TensorHandle<R>,
) -> Result<TensorHandle<R>, QRSetupError> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a.len() != 2 || shape_b.len() != 1 || shape_a[0] != shape_b[0] {
        return Err(QRSetupError::InvalidShape);
    }

    let m = shape_a[0];
    let n = shape_a[1];

    if m < n {
        return Err(QRSetupError::InvalidShape);
    }

    // 1. A = QR
    let (q, r) = strategy.launch::<R, E>(client, a)?;

    // 2. y = Q^T * b
    let y = TensorHandle::zeros(client, vec![m], a.dtype);
    let max_cube_dim = client.properties().hardware.max_cube_dim.0;
    let cd_q = CubeDim::new_1d(max_cube_dim.min(m as u32));
    let cc_q = calculate_cube_count_elemwise(client, m, cd_q);

    // Both BAHT and CGR now return Q^T
    unsafe {
        q_already_t_b_kernel::launch_unchecked::<E, R>(
            client,
            cc_q,
            cd_q,
            m as u32,
            q.clone().into_arg(),
            b.clone().into_arg(),
            y.clone().into_arg(),
        );
    }

    // 3. Rx = y (first n elements of y if m > n)
    let x = TensorHandle::zeros(client, vec![n], a.dtype);
    let is_col_major = 1u32;

    // For back substitution, we use a single cube since it's sequential
    unsafe {
        back_substitution_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            n as u32,
            m as u32,
            r.clone().into_arg(),
            y.clone().into_arg(),
            x.clone().into_arg(),
            is_col_major,
        );
    }

    Ok(x)
}
