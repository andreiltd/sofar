use std::io;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("The owls are not what they seem")]
    InternalError,
    #[error("Invalid data format")]
    InvalidFormat,
    #[error("Format is not supported")]
    UnsupportedFormat,
    #[error("Invalid attributes")]
    InvalidAttributes,
    #[error("Invalid dimensions")]
    InvalidDimensions,
    #[error("Invalid dimension list")]
    InvalidDimensionList,
    #[error("Invalid coordinate type")]
    InvalidCoordinateType,
    #[error("Invalid receiver position")]
    InvalidReceiverPositions,
    #[error("Emitters without ECI are not supported")]
    OnlyEmitterWithEciSupported,
    #[error("Delays without IR or MR are not supported")]
    OnlyDelaysWithIrOrMrSupported,
    #[error("Sources without MC are not supported")]
    OnlySourcesWithMcSupported,
    #[error("Sampling rates differ")]
    OnlyTheSameSamplingRateSupported,
}

impl Error {
    pub(crate) fn from_raw(err: i32) -> Error {
        use Error::*;

        match err {
            ffi::MYSOFA_INVALID_FORMAT => InvalidFormat,
            ffi::MYSOFA_UNSUPPORTED_FORMAT => UnsupportedFormat,
            ffi::MYSOFA_INVALID_ATTRIBUTES => InvalidAttributes,
            ffi::MYSOFA_INVALID_DIMENSIONS => InvalidDimensions,
            ffi::MYSOFA_INVALID_DIMENSION_LIST => InvalidDimensionList,
            ffi::MYSOFA_INVALID_COORDINATE_TYPE => InvalidCoordinateType,
            ffi::MYSOFA_INVALID_RECEIVER_POSITIONS => InvalidReceiverPositions,
            ffi::MYSOFA_ONLY_EMITTER_WITH_ECI_SUPPORTED => OnlyEmitterWithEciSupported,
            ffi::MYSOFA_ONLY_DELAYS_WITH_IR_OR_MR_SUPPORTED => OnlyDelaysWithIrOrMrSupported,
            ffi::MYSOFA_ONLY_SOURCES_WITH_MC_SUPPORTED => OnlySourcesWithMcSupported,
            ffi::MYSOFA_ONLY_THE_SAME_SAMPLING_RATE_SUPPORTED => OnlyTheSameSamplingRateSupported,
            ffi::MYSOFA_READ_ERROR => Io(io::Error::new(
                io::ErrorKind::NotFound,
                "Unable to read from file",
            )),
            ffi::MYSOFA_NO_MEMORY => Io(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Ran out of memory",
            )),
            _ => Error::InternalError,
        }
    }
}
