//! HDF5 file format parser for SOFA files.
//!
//! This module provides a pure Rust implementation of HDF5 parsing,
//! specifically tailored for reading SOFA (Spatially Oriented Format for Acoustics) files.

mod btree;
mod data_object;
mod fractal_heap;
mod gcol;
mod helpers;
mod ohdr_message;
mod parser;
mod super_block;

pub use data_object::{DataFormat, DataObject, DataSpace, DataType, GroupInfo, Record};
pub use fractal_heap::{Attribute, DirectoryEntry, FractalHeapData};
pub use gcol::GlobalHeap;
pub use parser::ParsedHdf;
pub use super_block::SuperBlock;

/// Parses an HDF5/SOFA file from bytes and returns the root DataObject.
///
/// # Errors
///
/// Returns an error if the input is not a valid HDF5 file or if parsing fails.
pub fn parse(input: &[u8]) -> Result<DataObject, winnow::error::ContextError> {
    parser::parse(input).map_err(|e| match e {
        winnow::error::ErrMode::Backtrack(e) | winnow::error::ErrMode::Cut(e) => e,
        winnow::error::ErrMode::Incomplete(_) => winnow::error::ContextError::new(),
    })
}

/// Parses an HDF5/SOFA file and returns a navigable structure that allows
/// parsing child objects.
///
/// # Errors
///
/// Returns an error if the input is not a valid HDF5 file or if parsing fails.
pub fn parse_with_children(input: &[u8]) -> Result<ParsedHdf<'_>, winnow::error::ContextError> {
    parser::parse_with_children(input).map_err(|e| match e {
        winnow::error::ErrMode::Backtrack(e) | winnow::error::ErrMode::Cut(e) => e,
        winnow::error::ErrMode::Incomplete(_) => winnow::error::ContextError::new(),
    })
}
