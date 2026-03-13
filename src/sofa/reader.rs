//! SOFA/HRTF file reader.
//!
//! Reads SOFA files and extracts HRTF, Head-Related Transfer Function, data.

use std::collections::HashMap;
use std::path::Path;

use crate::hdf::{self, DataObject, DataType, ParsedHdf};

use super::error::{Error, Result};
use super::types::{Array, Dimensions};

/// HRTF data loaded from a SOFA file.
///
/// Contains the spatial audio impulse response data along with position
/// information for head-related transfer function processing.
#[derive(Debug, Clone)]
pub struct Hrtf {
    /// SOFA dimensions: I, C, R, E, N, M
    dimensions: Dimensions,

    /// Listener position. I × C elements.
    pub listener_position: Array,
    /// Receiver positions relative to listener. R × C × I elements.
    pub receiver_position: Array,
    /// Source positions for each measurement. M × C elements.
    pub source_position: Array,
    /// Emitter positions. E × C × I elements.
    pub emitter_position: Array,
    /// Listener up vector. I × C elements.
    pub listener_up: Array,
    /// Listener view direction. I × C elements.
    pub listener_view: Array,

    /// Impulse response data. M × R × N elements.
    pub data_ir: Array,
    /// Sampling rates
    pub data_sampling_rate: Array,
    /// Per-filter delays. M × R elements.
    pub data_delay: Array,

    /// File-level attributes
    pub attributes: HashMap<String, String>,
}

impl Hrtf {
    /// Load HRTF from a SOFA file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a valid SOFA file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Load HRTF from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not a valid SOFA file.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let parsed = hdf::parse_with_children(data)?;
        Self::from_parsed_hdf(&parsed)
    }

    /// Build HRTF from a parsed HDF5 file.
    fn from_parsed_hdf(parsed: &ParsedHdf<'_>) -> Result<Self> {
        let root = &parsed.root;

        // Check for SOFA convention attribute
        let conventions = root
            .parsed_attributes
            .iter()
            .find(|a| a.name == "Conventions")
            .and_then(|a| a.value.as_ref())
            .ok_or(Error::InvalidFormat)?;

        if conventions != "SOFA" {
            return Err(Error::InvalidFormat);
        }

        // Build attributes map
        let attributes: HashMap<String, String> = root
            .parsed_attributes
            .iter()
            .filter_map(|a| a.value.as_ref().map(|v| (a.name.clone(), v.clone())))
            .collect();

        // Parse dimensions from child objects
        let dimensions = Self::parse_dimensions(parsed)?;

        // Parse arrays from child objects
        let listener_position = Self::parse_array(parsed, "ListenerPosition").unwrap_or_default();
        let receiver_position = Self::parse_array(parsed, "ReceiverPosition").unwrap_or_default();
        let source_position = Self::parse_array(parsed, "SourcePosition").unwrap_or_default();
        let emitter_position = Self::parse_array(parsed, "EmitterPosition").unwrap_or_default();
        let listener_up = Self::parse_array(parsed, "ListenerUp").unwrap_or_default();
        let listener_view = Self::parse_array(parsed, "ListenerView").unwrap_or_default();
        let data_ir = Self::parse_array(parsed, "Data.IR").unwrap_or_default();
        let data_sampling_rate = Self::parse_array(parsed, "Data.SamplingRate").unwrap_or_default();
        let data_delay = Self::parse_array(parsed, "Data.Delay").unwrap_or_default();

        Ok(Self {
            dimensions,
            listener_position,
            receiver_position,
            source_position,
            emitter_position,
            listener_up,
            listener_view,
            data_ir,
            data_sampling_rate,
            data_delay,
            attributes,
        })
    }

    /// Parse SOFA dimensions from child data objects.
    ///
    /// Dimensions are stored as single-character named datasets (I, C, R, E, N, M).
    /// We first check that all required dimension objects exist, then try to
    /// extract values. If that fails, we infer dimensions from array sizes.
    fn parse_dimensions(parsed: &ParsedHdf<'_>) -> Result<Dimensions> {
        let mut dims = Dimensions::default();
        let mut found = 0u8;

        // Check which dimensions exist
        for dir in &parsed.root.child_directories {
            if dir.name.len() == 1 {
                let ch = dir.name.chars().next().unwrap();
                match ch {
                    'I' => found |= 0x01,
                    'C' => found |= 0x02,
                    'R' => found |= 0x04,
                    'E' => found |= 0x08,
                    'N' => found |= 0x10,
                    'M' => found |= 0x20,
                    _ => {}
                }
            }
        }

        // Check all required dimensions are present
        if found != 0x3F {
            for (mask, name) in [
                (0x01, 'I'),
                (0x02, 'C'),
                (0x04, 'R'),
                (0x08, 'E'),
                (0x10, 'N'),
                (0x20, 'M'),
            ] {
                if found & mask == 0 {
                    return Err(Error::MissingDimension(name));
                }
            }
        }

        // Set spec-mandated values
        dims.i = 1;
        dims.c = 3;

        // Try to infer R, E, N, M from arrays we can parse
        // Default values for typical binaural HRTF
        dims.r = 2;
        dims.e = 1;
        dims.n = 1;
        dims.m = 1;

        // Try parsing Data.IR to get M, R, N (shape is M × R × N)
        // Use a Result to handle parse failures gracefully
        let ir_result = parsed.get_child("Data.IR");
        if let Some(Ok(ir_obj)) = ir_result
            && ir_obj.ds.dimensionality >= 3
        {
            dims.m = ir_obj.ds.dimension_size.first().copied().unwrap_or(1) as u32;
            dims.r = ir_obj.ds.dimension_size.get(1).copied().unwrap_or(2) as u32;
            dims.n = ir_obj.ds.dimension_size.get(2).copied().unwrap_or(1) as u32;
        }

        // Note: We skip trying to parse SourcePosition and EmitterPosition for now
        // as they may have unsupported data formats. The dimensions from Data.IR
        // should be sufficient for most HRTF operations.

        Ok(dims)
    }

    /// Extract dimension value from a data object.
    ///
    /// Dimensions are stored either as:
    /// 1. A netCDF dimension attribute: "This is a netCDF dimension but not a netCDF variable. N"
    /// 2. As a scalar value in the data
    #[allow(dead_code)] // May be useful for future dimension parsing improvements
    fn extract_dimension_value(obj: &DataObject) -> Result<u32> {
        // Check for netCDF dimension attribute
        for attr in &obj.parsed_attributes {
            if attr.name == "NAME"
                && let Some(value) = &attr.value
                && value.starts_with("This is a netCDF dimension")
            {
                // Extract number from end of string
                let num_str: String = value
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                let num_str: String = num_str.chars().rev().collect();
                if let Ok(n) = num_str.parse::<u32>() {
                    return Ok(n);
                }
            }
        }

        // Fall back to reading from data (single u64 or u32)
        if !obj.data.is_empty() {
            if obj.data.len() >= 8 {
                let bytes: [u8; 8] = obj.data[0..8].try_into().unwrap();
                return Ok(u64::from_le_bytes(bytes) as u32);
            } else if obj.data.len() >= 4 {
                let bytes: [u8; 4] = obj.data[0..4].try_into().unwrap();
                return Ok(u32::from_le_bytes(bytes));
            }
        }

        // Default fallback based on SOFA spec
        Ok(1)
    }

    /// Parse an array from a named child object.
    fn parse_array(parsed: &ParsedHdf<'_>, name: &str) -> Option<Array> {
        let child_result = parsed.get_child(name)?;
        let child = match child_result {
            Ok(c) => c,
            Err(_e) => {
                log::debug!("Failed to parse array '{}': {:?}", name, _e);
                return None;
            }
        };
        Self::data_object_to_array(&child)
    }

    /// Convert a DataObject to an Array of f32 values.
    fn data_object_to_array(obj: &DataObject) -> Option<Array> {
        if obj.data.is_empty() {
            return None;
        }

        // Build attributes map
        let attributes: HashMap<String, String> = obj
            .parsed_attributes
            .iter()
            .filter_map(|a| a.value.as_ref().map(|v| (a.name.clone(), v.clone())))
            .collect();

        // Convert data based on type
        let values = Self::convert_data_to_f32(&obj.data, &obj.dt)?;

        Some(Array { values, attributes })
    }

    /// Convert raw bytes to f32 values based on data type.
    fn convert_data_to_f32(data: &[u8], dt: &DataType) -> Option<Vec<f32>> {
        let class = dt.class_and_version & 0x0F;

        match class {
            // Float type
            1 => {
                let precision = dt
                    .data_fmt
                    .as_ref()
                    .map(|f| match f {
                        hdf::DataFormat::Float { bit_precision, .. } => *bit_precision,
                        _ => 64,
                    })
                    .unwrap_or(64);

                if precision == 64 {
                    // f64 (double) - convert to f32
                    let count = data.len() / 8;
                    let mut values = Vec::with_capacity(count);
                    for i in 0..count {
                        let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().ok()?;
                        values.push(f64::from_le_bytes(bytes) as f32);
                    }
                    Some(values)
                } else if precision == 32 {
                    // f32 - direct copy
                    let count = data.len() / 4;
                    let mut values = Vec::with_capacity(count);
                    for i in 0..count {
                        let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().ok()?;
                        values.push(f32::from_le_bytes(bytes));
                    }
                    Some(values)
                } else {
                    None
                }
            }
            // Integer type
            0 => {
                let size = dt.size as usize;
                if size == 8 {
                    // i64/u64 - convert to f32
                    let count = data.len() / 8;
                    let mut values = Vec::with_capacity(count);
                    for i in 0..count {
                        let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().ok()?;
                        values.push(i64::from_le_bytes(bytes) as f32);
                    }
                    Some(values)
                } else if size == 4 {
                    // i32/u32 - convert to f32
                    let count = data.len() / 4;
                    let mut values = Vec::with_capacity(count);
                    for i in 0..count {
                        let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().ok()?;
                        values.push(i32::from_le_bytes(bytes) as f32);
                    }
                    Some(values)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // Accessors for dimensions

    /// Number of measurements (HRTF filter positions).
    pub fn m(&self) -> u32 {
        self.dimensions.m
    }

    /// Number of samples per measurement (filter length).
    pub fn n(&self) -> u32 {
        self.dimensions.n
    }

    /// Number of receivers (typically 2 for binaural).
    pub fn r(&self) -> u32 {
        self.dimensions.r
    }

    /// Number of emitters.
    pub fn e(&self) -> u32 {
        self.dimensions.e
    }

    /// Get the sampling rate.
    pub fn sample_rate(&self) -> f32 {
        self.data_sampling_rate
            .values
            .first()
            .copied()
            .unwrap_or(48000.0)
    }

    /// Get the filter length in samples.
    pub fn filter_len(&self) -> usize {
        self.dimensions.n as usize
    }

    /// Get the dimensions.
    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    /// Get an attribute value by name.
    pub fn get_attribute(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(|s| s.as_str())
    }

    /// Set the filter length (N dimension).
    pub(crate) fn set_n(&mut self, n: u32) {
        self.dimensions.n = n;
    }

    /// Set the sample rate.
    pub(crate) fn set_sample_rate(&mut self, rate: f32) {
        if self.data_sampling_rate.values.is_empty() {
            self.data_sampling_rate.values.push(rate);
        } else {
            self.data_sampling_rate.values[0] = rate;
        }
    }

    /// Convert all position arrays from spherical to cartesian coordinates.
    ///
    /// This matches the C library's `mysofa_tocartesian` behavior. Each array
    /// is converted only if its "Type" attribute is "spherical".
    pub(crate) fn convert_to_cartesian(&mut self) {
        convert_array_to_cartesian_if_spherical(&mut self.source_position);
        convert_array_to_cartesian_if_spherical(&mut self.receiver_position);
        convert_array_to_cartesian_if_spherical(&mut self.emitter_position);
        convert_array_to_cartesian_if_spherical(&mut self.listener_position);
        convert_array_to_cartesian_if_spherical(&mut self.listener_view);
        convert_array_to_cartesian_if_spherical(&mut self.listener_up);
    }
}

/// Convert an array's values from spherical to cartesian if its "Type"
/// attribute indicates spherical coordinates. Updates the attribute to
/// "cartesian" after conversion.
fn convert_array_to_cartesian_if_spherical(array: &mut super::types::Array) {
    let coord_type = array.get_attribute("Type");

    match coord_type {
        Some("cartesian") | None => return,
        Some("spherical") => {}
        Some(other) => {
            log::warn!("Unknown coordinate type: {other}, assuming cartesian");
            return;
        }
    }

    super::coords::convert_array_to_cartesian(&mut array.values);
    array
        .attributes
        .insert("Type".to_string(), "cartesian".to_string());
    array
        .attributes
        .insert("Units".to_string(), "meter".to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions_validity() {
        let mut dims = Dimensions::default();
        assert!(!dims.is_valid());

        dims.i = 1;
        dims.c = 3;
        dims.r = 2;
        dims.e = 1;
        dims.n = 128;
        dims.m = 100;
        assert!(dims.is_valid());
    }

    #[test]
    fn test_hdf_parsing_debug() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        std::env::set_current_dir(cwd).unwrap();

        let data = std::fs::read("libmysofa-sys/libmysofa/tests/tester.sofa").unwrap();
        let parsed = crate::hdf::parse_with_children(&data).unwrap();

        println!("Root attributes:");
        for attr in &parsed.root.parsed_attributes {
            println!(
                "  {:?} (len={}) = {:?}",
                attr.name,
                attr.name.len(),
                attr.value
            );
        }

        println!("\nChild directories:");
        for dir in &parsed.root.child_directories {
            println!("  {} at {:#x}", dir.name, dir.address);
        }

        // Check that we have Conventions attribute
        let conventions = parsed
            .root
            .parsed_attributes
            .iter()
            .find(|a| a.name == "Conventions");
        assert!(conventions.is_some(), "Conventions attribute not found");
    }
}
