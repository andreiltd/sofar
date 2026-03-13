//! SOFA file reader and processor for Spatially Oriented Format for Acoustics.
//!
//! This module provides pure Rust implementation for reading and processing
//! SOFA files containing HRTF, or Head-Related Transfer Function, data.
//!
//! This module is internal. Use [`crate::reader`] for the public API.

pub mod coords;
mod error;
mod interpolate;
mod kdtree;
mod lookup;
mod loudness;
mod neighbors;
mod reader;
mod resample;
mod types;
mod validate;

pub use error::Error;
pub use interpolate::{InterpolatedFilter, get_filter_nointerp, interpolate};
pub use lookup::Lookup;
pub use loudness::normalize;
pub use neighbors::Neighborhood;
pub use reader::Hrtf;
pub use resample::resample;
pub use validate::validate;
