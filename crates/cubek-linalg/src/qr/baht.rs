// Blocked Accelerated Householder QR Decomposition
// References:
//   https://github.com/janverschelde/PHCpack/blob/master/src/GPU/Matrices/dbl_baqr_kernels.cu
//   https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/5/462/files/2016/08/Kerr_Campbell_Richards_QRD_on_GPUs.pdf
//
// Algorithm:
//   For each block k:
//     1. Compute Householder vectors for the tile columns, reducing R_k,k.
//     2. Build WY^T such that I + WY^T = H_1*...*H_s (compact WY representation).
//     3. Update Q: Q = Q * (I + WY^T).

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

// ---------------------------------------------------------------------------
// Householder reflector for a single column.
// Each thread ji handles element ji of the subdiagonal part.
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn householder_kernel<F: Float>(
    r: &Array<F>,
    r_offset: u32,
    dim: u32,
    v: &mut Array<F>,
    beta: &mut Array<F>,
    beta_offset: u32,
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let tdx_usize = tdx as usize;
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);

    let mut product = SharedMemory::<F>::new(shared_size);

    let mut local_prd = zero;
    let mut j = tdx;
    let cube_dim_x = CUBE_DIM_X;
    let cube_dim_x_usize = cube_dim_x as usize;
    while j < dim {
        let val = r[(r_offset + j + 1) as usize];
        local_prd += val * val;
        j += cube_dim_x;
    }
    product[tdx_usize] = local_prd;
    sync_cube();

    let mut pow2 = 1u32;
    while pow2 < cube_dim_x {
        if tdx_usize.is_multiple_of((pow2 as usize) * 2) && (tdx_usize + (pow2 as usize) < cube_dim_x_usize) {
            product[tdx_usize] = product[tdx_usize] + product[tdx_usize + pow2 as usize];
        }
        pow2 *= 2;
        sync_cube();
    }

    let is_zero = product[0] == zero;
    if !is_zero {
        if tdx == 0 {
            let mu = F::sqrt(r[r_offset as usize] * r[r_offset as usize] + product[0]);
            let v0 = if r[r_offset as usize] <= zero {
                r[r_offset as usize] - mu
            } else {
                -product[0] / (r[r_offset as usize] + mu)
            };
            let v0sq = v0 * v0;
            beta[beta_offset as usize] = two * v0sq / (product[0] + v0sq);
            product[0] = v0;
        }
        sync_cube();

        let b = beta[beta_offset as usize];
        let mut j2 = tdx;
        while j2 < dim {
            if b != zero {
                v[(j2 + 1) as usize] = r[(r_offset + j2 + 1) as usize] / product[0];
            } else {
                v[(j2 + 1) as usize] = r[(r_offset + j2 + 1) as usize];
            }
            j2 += cube_dim_x;
        }
    } else {
        let mut j2 = tdx;
        while j2 < dim {
            v[(j2 + 1) as usize] = r[(r_offset + j2 + 1) as usize];
            j2 += cube_dim_x;
        }
        if tdx == 0 {
            beta[beta_offset as usize] = zero;
        }
    }

    if tdx == 0 {
        v[0] = one;
    }
}

// ---------------------------------------------------------------------------
// Apply H = I - beta*v*v^T to columns of R from the left.
// One thread per column; all columns from col_offset to cols-1.
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn left_update_r_kernel<F: Float + CubeElement>(
    rows: u32,
    cols: u32,
    col_offset: u32,
    r: &mut Tensor<F>,
    v: &Array<F>,
    beta: F,
    #[comptime] shared_size: usize,
) {
    let tdx = UNIT_POS_X;
    let col_idx = ABSOLUTE_POS_X;
    let n_update = cols - col_offset;
    let in_range = col_idx < n_update;
    let r_base = col_offset * rows + col_offset;
    let n_rows = rows - col_offset;

    let mut shv = SharedMemory::<F>::new(shared_size);
    let mut load_idx = tdx as usize;
    let cube_dim_x = CUBE_DIM_X as usize;
    while load_idx < shared_size {
        shv[load_idx] = v[load_idx];
        load_idx += cube_dim_x;
    }
    sync_cube();

    let mut w = F::from_int(0);
    if in_range {
        for i in 0u32..n_rows {
            let r_idx = (r_base + i + col_idx * rows) as usize;
            w += r[r_idx] * shv[i as usize];
        }
        w *= beta;
    }
    sync_cube();

    if in_range {
        for i in 0u32..n_rows {
            let r_idx = (r_base + i + col_idx * rows) as usize;
            r[r_idx] -= shv[i as usize] * w;
        }
    }
}

// ---------------------------------------------------------------------------
// w_0 = -beta * v   (first WY^T column)
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn init_w_kernel<F: Float + CubeElement>(n: u32, beta: F, v: &Array<F>, w: &mut Array<F>) {
    let idx = ABSOLUTE_POS;
    if idx < n as usize {
        w[idx] = -beta * v[idx];
    }
}

// ---------------------------------------------------------------------------
// WY^T = outer(w, v)   (initialization from first reflector)
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn init_wyt_kernel<F: Float>(rowdim: u32, n: u32, v: &Array<F>, w: &Array<F>, wyt: &mut Array<F>) {
    let idx = ABSOLUTE_POS;
    if idx < (n * n) as usize {
        let row = idx / n as usize;
        let col = idx % n as usize;
        // line is 0 here, so v has no leading zeros
        wyt[row * rowdim as usize + col] = w[row] * v[col];
    }
}

// ---------------------------------------------------------------------------
// WY^T += outer(w_j, v_j)
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn update_wyt_kernel<F: Float>(rowdim: u32, line: u32, n: u32, v: &Array<F>, w: &Array<F>, wyt: &mut Array<F>) {
    let idx = ABSOLUTE_POS;
    if idx < (n * n) as usize {
        let row = idx / n as usize;
        let col = idx % n as usize;
        let v_val = if col < line as usize { F::from_int(0) } else { v[col - line as usize] };
        wyt[row * rowdim as usize + col] += w[row] * v_val;
    }
}

// ---------------------------------------------------------------------------
// w_j = -beta_j * (v + WY^T * v)
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn next_w_kernel<F: Float + CubeElement>(
    rowdim: u32,
    line: u32,
    n: u32,
    beta: F,
    v: &Array<F>,
    wyt: &Array<F>,
    w: &mut Array<F>,
    #[comptime] shared_size: usize,
) {
    let idx = ABSOLUTE_POS;
    let tdx = UNIT_POS_X;
    let mut shv = SharedMemory::<F>::new(shared_size);
    let mut load_idx = tdx as usize;
    let cube_dim_x = CUBE_DIM_X as usize;
    while load_idx < shared_size {
        if load_idx < line as usize {
            shv[load_idx] = F::from_int(0);
        } else {
            shv[load_idx] = v[load_idx - line as usize];
        }
        load_idx += cube_dim_x;
    }
    sync_cube();
    if idx < n as usize {
        let mut acc = shv[idx];
        for k in 0..n as usize {
            acc += wyt[idx * rowdim as usize + k] * shv[k];
        }
        w[idx] = -beta * acc;
    }
}

// ---------------------------------------------------------------------------
// Q[:,col_offset:] += Q[:,col_offset:] * WY^T
// One thread per (q_row, wyt_col) in the update block.
// ---------------------------------------------------------------------------
#[cube(launch_unchecked)]
fn update_q_kernel<F: Float>(
    q_dim: u32,
    rowdim: u32,
    col_offset: u32,
    q: &mut Tensor<F>,
    q_old: &Tensor<F>,
    wyt: &Array<F>,
    #[comptime] _shared_size: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < (q_dim * rowdim) as usize {
        let q_row = idx / rowdim as usize;
        let col = idx % rowdim as usize;

        let mut acc = F::from_int(0);
        for k in 0..rowdim as usize {
            acc += q_old[q_row * q_dim as usize + col_offset as usize + k] * wyt[k * rowdim as usize + col];
        }
        q[q_row * q_dim as usize + col_offset as usize + col] = q_old[q_row * q_dim as usize + col_offset as usize + col] + acc;
    }
}

// ---------------------------------------------------------------------------
// Public entry: blocked Householder QR
// ---------------------------------------------------------------------------
pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R>,
    q: &TensorHandle<R>,
    r: &TensorHandle<R>,
) {
    let rows = r.shape()[0] as u32;
    let cols = r.shape()[1] as u32;
    let tile = client.properties().hardware.plane_size_min.min(cols);
    let num_tiles = cols.div_ceil(tile);
    let dtype = E::as_type_native_unchecked();

    let v = TensorHandle::<R>::zeros(client, vec![rows as usize], dtype);
    let w = TensorHandle::<R>::zeros(client, vec![rows as usize], dtype);
    let beta = TensorHandle::<R>::zeros(client, vec![tile as usize], dtype);
    let wyt = TensorHandle::<R>::zeros(client, vec![(rows * rows) as usize], dtype);
    let max_cube_dim = client.properties().hardware.max_cube_dim.0;

    for k in 0..num_tiles {
        let col_start = k * tile;
        let rowdim = rows - col_start;
        let n = rowdim as usize;

        for line in 0..tile {
            let col = col_start + line;
            if col >= cols {
                break;
            }
            let rows_below = rows - col - 1;
            if rows_below == 0 {
                continue;
            }

            // Compute Householder reflector for this column
            let row_offset = (col * (rows + 1)) as u64;
            let shared_size = max_cube_dim as usize;
            let mut pow2 = 1u32;
            while pow2 < rows_below && pow2 < max_cube_dim {
                pow2 *= 2;
            }
            let cd = CubeDim::new_1d(pow2);
            let cc = calculate_cube_count_elemwise(client, 1usize, cd);
            unsafe {
                householder_kernel::launch_unchecked::<E, R>(
                    client,
                    cc,
                    cd,
                    ArrayArg::from_raw_parts(r.handle.clone(), (rows * cols) as usize),
                    row_offset as u32,
                    rows_below,
                    ArrayArg::from_raw_parts(v.handle.clone(), rows_below as usize + 1),
                    ArrayArg::from_raw_parts(beta.handle.clone(), tile as usize),
                    line,
                    shared_size,
                );
            }

            let beta_bytes = client.read_one(beta.handle.clone()).unwrap();
            let beta_vals = E::from_bytes(&beta_bytes);
            if beta_vals[line as usize] == E::from_int(0) {
                continue;
            }
            let beta_val = beta_vals[line as usize];

            // Apply to R from the left
            let n_upd = (cols - col) as usize;
            let cd = CubeDim::new_1d((n_upd as u32).min(max_cube_dim));
            let cc = calculate_cube_count_elemwise(client, n_upd, cd);
            unsafe {
                left_update_r_kernel::launch_unchecked::<E, R>(
                    client,
                    cc,
                    cd,
                    rows,
                    cols,
                    col,
                    r.clone().into_arg(),
                    ArrayArg::from_raw_parts(v.handle.clone(), (rows - col) as usize),
                    beta_val,
                    (rows - col) as usize,
                );
            }

            // Build WY^T
            let total = n * n;
            let cd2 = CubeDim::new_1d(max_cube_dim.min(total as u32));
            let cc2 = calculate_cube_count_elemwise(client, total, cd2);
            if line == 0 {
                let cd_w = CubeDim::new_1d(rowdim.min(max_cube_dim));
                let cc_w = calculate_cube_count_elemwise(client, n, cd_w);
                unsafe {
                    init_w_kernel::launch_unchecked::<E, R>(
                        client,
                        cc_w,
                        cd_w,
                        rowdim,
                        beta_val,
                        ArrayArg::from_raw_parts(v.handle.clone(), n),
                        ArrayArg::from_raw_parts(w.handle.clone(), n),
                    );
                    init_wyt_kernel::launch_unchecked::<E, R>(
                        client,
                        cc2,
                        cd2,
                        rowdim,
                        rowdim,
                        ArrayArg::from_raw_parts(v.handle.clone(), n),
                        ArrayArg::from_raw_parts(w.handle.clone(), n),
                        ArrayArg::from_raw_parts(wyt.handle.clone(), total),
                    );
                }
            } else {
                let cd_w = CubeDim::new_1d(rowdim.min(max_cube_dim));
                let cc_w = calculate_cube_count_elemwise(client, n, cd_w);
                unsafe {
                    next_w_kernel::launch_unchecked::<E, R>(
                        client,
                        cc_w,
                        cd_w,
                        rowdim,
                        line,
                        rowdim,
                        beta_val,
                        ArrayArg::from_raw_parts(v.handle.clone(), n),
                        ArrayArg::from_raw_parts(wyt.handle.clone(), total),
                        ArrayArg::from_raw_parts(w.handle.clone(), n),
                        n,
                    );
                    update_wyt_kernel::launch_unchecked::<E, R>(
                        client,
                        cc2,
                        cd2,
                        rowdim,
                        line,
                        rowdim,
                        ArrayArg::from_raw_parts(v.handle.clone(), n),
                        ArrayArg::from_raw_parts(w.handle.clone(), n),
                        ArrayArg::from_raw_parts(wyt.handle.clone(), total),
                    );
                }
            }
        }

        // Update Q: Q[:,col_start:] += Q[:,col_start:] * WY^T
        let q_old = TensorHandle::<R>::new_contiguous(
            q.shape().to_vec(),
            client.create_from_slice(&client.read_one(q.handle.clone()).unwrap()),
            q.dtype,
        );

        let q_dim = rows; // Q is square
        let total_q = (q_dim * rowdim) as usize;
        let cd = CubeDim::new_1d(max_cube_dim.min(total_q as u32));
        let cc = calculate_cube_count_elemwise(client, total_q, cd);
        unsafe {
            update_q_kernel::launch_unchecked::<E, R>(
                client,
                cc,
                cd,
                q_dim,
                rowdim,
                col_start,
                q.clone().into_arg(),
                q_old.clone().into_arg(),
                ArrayArg::from_raw_parts(wyt.handle.clone(), n * n),
                n,
            );
        }
    }
}
