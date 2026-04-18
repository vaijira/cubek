mod cgr;
pub mod house;
pub mod solve;
pub mod utils;

#[macro_export]
macro_rules! testgen_qr_cgr {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::cgr::test_qr_cgr::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::cgr::test_qr_cgr::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::cgr::test_qr_cgr::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::cgr::test_qr_cgr::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::cgr::test_qr_cgr_rect::<FloatT>(517, 157);
            }

    };
    ([$($float:ident),*]) => {
        mod cgr {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_cgr!($float);
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_qr_house {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                crate::suite::house::test_qr_baht::<FloatT>(3);
            }

            #[test]
            pub fn test_small() {
                crate::suite::house::test_qr_baht::<FloatT>(47);
            }

            #[test]
            pub fn test_medium() {
                crate::suite::house::test_qr_baht::<FloatT>(157);
            }

            #[test]
            pub fn test_big() {
                crate::suite::house::test_qr_baht::<FloatT>(517);
            }

            #[test]
            pub fn test_rect() {
                crate::suite::house::test_qr_baht_rect::<FloatT>(517, 157);
            }

    };
    ([$($float:ident),*]) => {
        mod house {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_qr_house!($float);
                })*
            }
        }
    };
}

mod test_cgr {
    testgen_qr_cgr!([f32, f64]);
}

mod test_house {
    testgen_qr_house!([f32, f64]);
}

#[macro_export]
macro_rules! testgen_solve {
   ($float:ident) => {
            pub type FloatT = $float;

            #[test]
            pub fn test_solve_square_house() {
                crate::suite::solve::test_solve_square::<FloatT>(16, cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder);
            }

            #[test]
            pub fn test_solve_square_givens() {
                crate::suite::solve::test_solve_square::<FloatT>(16, cubek_linalg::QRStrategy::CommonGivensRotations);
            }

            #[test]
            pub fn test_solve_rect_house() {
                crate::suite::solve::test_solve_rect::<FloatT>(32, 16, cubek_linalg::QRStrategy::BlockedAcceleratedHouseHolder);
            }

            #[test]
            pub fn test_solve_rect_givens() {
                crate::suite::solve::test_solve_rect::<FloatT>(32, 16, cubek_linalg::QRStrategy::CommonGivensRotations);
            }
    };
    ([$($float:ident),*]) => {
        mod solve {
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    $crate::testgen_solve!($float);
                })*
            }
        }
    };
}

mod test_solve {
    testgen_solve!([f32, f64]);
}
 