//! Core data types for SOFA files.
//!
//! These types represent SOFA-specific structures built on top of the HDF5 parser.

use std::collections::HashMap;

/// SOFA dimensions as defined in AES69 standard.
///
/// - `I`: Singleton dimension, always 1
/// - `C`: Coordinate triplet, always 3
/// - `R`: Number of receivers, i.e. microphone capsules
/// - `E`: Number of emitters, i.e. sound sources
/// - `N`: Number of samples per measurement, i.e. the filter length
/// - `M`: Number of measurements, the total HRTF filters
#[derive(Debug, Clone, Copy, Default)]
pub struct Dimensions {
    /// Singleton dimension, always 1
    pub i: u32,
    /// Coordinate triplet, always 3
    pub c: u32,
    /// Number of receivers
    pub r: u32,
    /// Number of emitters
    pub e: u32,
    /// Number of samples per measurement, i.e. the filter length
    pub n: u32,
    /// Number of measurements
    pub m: u32,
}

impl Dimensions {
    /// Check if all required dimensions are present and valid.
    pub fn is_valid(&self) -> bool {
        self.i == 1 && self.c == 3 && self.r > 0 && self.e > 0 && self.n > 0 && self.m > 0
    }
}

/// A multidimensional array of float values with associated attributes.
///
/// This is the SOFA-level representation of data arrays like SourcePosition,
/// DataIR, etc. The values are stored as f32 for efficient processing.
#[derive(Debug, Clone, Default)]
pub struct Array {
    /// The actual float values, flattened in row-major order
    pub values: Vec<f32>,
    /// Associated attributes (e.g., "Type" for coordinate type)
    pub attributes: HashMap<String, String>,
}

impl Array {
    /// Create a new empty array.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an array with the given values.
    pub fn from_values(values: Vec<f32>) -> Self {
        Self {
            values,
            attributes: HashMap::new(),
        }
    }

    /// Get an attribute value by name.
    pub fn get_attribute(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(|s| s.as_str())
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.values.len()
    }
}
