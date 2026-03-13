//! Error types for SOFA file processing.

/// Error type for SOFA operations.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Invalid SOFA format: missing 'Conventions: SOFA' attribute")]
    InvalidFormat,
    #[error("Missing required dimension: {0}")]
    MissingDimension(char),
    #[error("Invalid dimension {name}: got {value}, expected {expected}")]
    InvalidDimension {
        name: char,
        value: u32,
        expected: u32,
    },
    #[error("Missing required array: {0}")]
    MissingArray(&'static str),
    #[error("Invalid array size for {name}: expected {expected}, got {actual}")]
    InvalidArraySize {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Invalid attribute {name}: expected {expected}")]
    InvalidAttribute {
        name: &'static str,
        expected: &'static str,
    },
    #[error("Unsupported data type")]
    UnsupportedDataType,
}

impl From<winnow::error::ContextError> for Error {
    fn from(e: winnow::error::ContextError) -> Self {
        Error::Parse(format!("{:?}", e))
    }
}

/// Result type for SOFA operations.
pub type Result<T> = std::result::Result<T, Error>;
