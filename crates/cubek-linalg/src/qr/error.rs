use thiserror::Error;

/// Errors that can occur when trying to launch QR decomposition.
#[derive(Debug, Error, PartialEq, Eq, Clone, Hash)]
pub enum QRSetupError {
    /// The input should be a matrix where m should be greater or equal to n.
    #[error("The input should be a matrix where m should be greater or equal to n.")]
    InvalidShape,
}
