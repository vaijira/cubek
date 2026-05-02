use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

use crate::qr::QRSetupError;
use crate::qr::{baht, baht_tsqr, cgr};

type QRTuple<R> = (TensorHandle<R>, TensorHandle<R>);

/// Define the strategy to use when calling for a QR decomposition.
#[derive(Debug, Clone, Default)]
pub enum QRStrategy {
    /// Numerically stable, dense matrix using blocked Householder reflectors.
    BlockedAcceleratedHouseHolder,
    /// TSQR-inspired: entire panel in one fused kernel per tile (min dispatches).
    BahtTsqr,
    /// Performs the QR decomposition using Givens rotations.
    /// Better for sparse matrices and less numerically stable than Householder transformations.
    CommonGivensRotations,
    /// Automatically choose the best strategy.
    #[default]
    Auto,
}

fn initialize<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
) -> Result<QRTuple<R>, QRSetupError> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] < shape[1] {
        return Err(QRSetupError::InvalidShape);
    }
    let dtype = a.dtype;
    let (m, _n) = (shape[0], shape[1]);

    // Allocate Q as identity (col-major [rows, rows]).
    // IMPORTANT: TensorHandle::zeros may return a GPU-padded buffer (e.g. pitch 4
    // for a 3-row matrix).  The BAHT kernels index Q^T with flat `col*rows+row`
    // arithmetic that assumes NO pitch padding.  We therefore build the identity
    // CPU-side and upload it via create_from_slice, which always produces a tight
    // (non-padded) buffer.
    let q_shape = vec![m, m];
    let elem_size = dtype.size();
    let mut identity_bytes = vec![0u8; m * m * elem_size];
    // Write 1.0 at each diagonal position in col-major order (col*m+row = col*m+col for diag).
    for i in 0..m {
        let byte_offset = (i * m + i) * elem_size;
        // Write the float 1.0 in the element's native byte representation.
        // We do this by writing the bytes of F::from_int(1) — but since we only have
        // a generic `StorageType` here, use the known IEEE 754 bit pattern for 1.0:
        //   f32: 0x3F80_0000  f64: 0x3FF0_0000_0000_0000
        match elem_size {
            4 => {
                let one: u32 = 0x3F80_0000;
                identity_bytes[byte_offset..byte_offset + 4].copy_from_slice(&one.to_le_bytes());
            }
            8 => {
                let one: u64 = 0x3FF0_0000_0000_0000;
                identity_bytes[byte_offset..byte_offset + 8].copy_from_slice(&one.to_le_bytes());
            }
            _ => {
                panic!("Unsupported element size {elem_size} for identity init");
            }
        }
    }
    let q_handle = client.create_from_slice(&identity_bytes);
    let q = TensorHandle::<R>::new(q_handle, q_shape, vec![1, m], dtype);

    // Build R as a tight col-major copy of A.  We must NOT use into_contiguous here
    // because that produces row-major output, which would make R's bytes inconsistent
    // with the col-major strides [1, m] the BAHT/CGR kernels expect.
    //
    // Instead: read A via its own strides (using into_contiguous which copies elements
    // in logical order from the source) then re-declare the output as col-major — BUT
    // only if A is already col-major.  The simplest safe approach: read A's raw bytes
    // directly (the test uses create_from_slice so no padding) and declare [1, m].
    let a_bytes = client.read_one(a.handle.clone()).unwrap();
    let a_handle = client.create_from_slice(&a_bytes);
    let a_strides = a.strides().to_vec(); // preserve the original strides
    let r = TensorHandle::<R>::new(a_handle, shape.to_vec(), a_strides, dtype);

    Ok((q, r))
}

impl QRStrategy {
    /// It launches a QR decomposition over a m x n matrix a.
    ///
    /// Specify a strategy for the QR decomposition, the client and the matrix a to decompose.
    /// In case of success it will return a tuple with the matrix Q and the matrix R in this order.
    pub fn launch<R: Runtime, EG: Float + CubeElement>(
        &self,
        client: &ComputeClient<R>,
        a: &TensorHandle<R>,
    ) -> Result<QRTuple<R>, QRSetupError> {
        let (q, r) = initialize::<R, EG>(client, a)?;

        match self {
            QRStrategy::BlockedAcceleratedHouseHolder => {
                baht::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::BahtTsqr => {
                baht_tsqr::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::CommonGivensRotations => {
                cgr::launch::<R, EG>(client, &q, &r);
            }
            QRStrategy::Auto => {
                baht::launch::<R, EG>(client, &q, &r);
            }
        };

        Ok((q, r))
    }

    /// Solves the system of equations Ax = b using the QR decomposition.
    pub fn solve<R: Runtime, EG: Float + CubeElement>(
        &self,
        client: &ComputeClient<R>,
        a: &TensorHandle<R>,
        b: &TensorHandle<R>,
    ) -> Result<TensorHandle<R>, QRSetupError> {
        crate::qr::solve::solve::<R, EG>(self, client, a, b)
    }
}

/// It launches a QR decomposition over a m x n matrix a.
///
/// Specify a strategy for the QR decomposition, the client and the matrix a to decompose.
/// In case of success it will return a tuple with the matrix Q and the matrix R in this order.
pub fn launch<R: Runtime, EG: Float + CubeElement>(
    strategy: &QRStrategy,
    client: &ComputeClient<R>,
    a: &TensorHandle<R>,
) -> Result<QRTuple<R>, QRSetupError> {
    strategy.launch::<R, EG>(client, a)
}
