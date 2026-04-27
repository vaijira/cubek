mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_convolution_accelerated {
    () => {
        mod conv2d_accelerated {
            use super::*;
            use cubek_std::tile::Strided;

            mod cmma {
                use super::*;
                type TMM = cubek_matmul::components::tile_matmul::cmma::CmmaMatmul;

                $crate::testgen_convolution_accelerated_algorithm!();
            }

            mod mma {
                use super::*;
                type TMM = cubek_matmul::components::tile_matmul::mma::MmaMatmul;

                $crate::testgen_convolution_accelerated_algorithm!();
            }
        }
    };
}
