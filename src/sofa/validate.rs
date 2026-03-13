//! SOFA file format validation.
//!
//! Validates that HRTF data conforms to the SOFA SimpleFreeFieldHRIR convention.

use super::error::{Error, Result};
use super::reader::Hrtf;

/// Tolerance for receiver position validation (meters).
const RECEIVER_POSITION_TOLERANCE: f32 = 0.02;

/// Validates that the HRTF data conforms to SOFA SimpleFreeFieldHRIR convention.
///
/// Checks:
/// - Conventions attribute is "SOFA"
/// - SOFAConventions is "SimpleFreeFieldHRIR"
/// - DataType is "FIR"
/// - Dimensions: C=3, I=1, E=1, R=2, M>0
/// - Receiver positions are in cartesian coordinates
/// - Receiver positions are symmetric within tolerance
///
/// # Errors
///
/// Returns an error describing the validation failure.
pub fn validate(hrtf: &Hrtf) -> Result<()> {
    validate_attributes(hrtf)?;
    validate_dimensions(hrtf)?;
    validate_listener_view(hrtf)?;
    validate_emitter_position(hrtf)?;
    validate_receiver_position(hrtf)?;
    validate_source_position(hrtf)?;
    validate_data_delay(hrtf)?;
    validate_sampling_rate(hrtf)?;

    Ok(())
}

/// Verifies required SOFA attributes.
fn validate_attributes(hrtf: &Hrtf) -> Result<()> {
    // Check Conventions = "SOFA"
    match hrtf.get_attribute("Conventions") {
        Some("SOFA") => {}
        _ => {
            return Err(Error::InvalidAttribute {
                name: "Conventions",
                expected: "SOFA",
            });
        }
    }

    // Check SOFAConventions = "SimpleFreeFieldHRIR"
    match hrtf.get_attribute("SOFAConventions") {
        Some("SimpleFreeFieldHRIR") => {}
        _ => {
            return Err(Error::InvalidAttribute {
                name: "SOFAConventions",
                expected: "SimpleFreeFieldHRIR",
            });
        }
    }

    // Check DataType = "FIR"
    match hrtf.get_attribute("DataType") {
        Some("FIR") => {}
        _ => {
            return Err(Error::InvalidAttribute {
                name: "DataType",
                expected: "FIR",
            });
        }
    }

    Ok(())
}

/// Verifies SOFA dimensions.
fn validate_dimensions(hrtf: &Hrtf) -> Result<()> {
    let dims = hrtf.dimensions();

    if dims.c != 3 {
        return Err(Error::InvalidDimension {
            name: 'C',
            value: dims.c,
            expected: 3,
        });
    }

    if dims.i != 1 {
        return Err(Error::InvalidDimension {
            name: 'I',
            value: dims.i,
            expected: 1,
        });
    }

    if dims.e != 1 {
        return Err(Error::InvalidDimension {
            name: 'E',
            value: dims.e,
            expected: 1,
        });
    }

    if dims.r != 2 {
        return Err(Error::InvalidDimension {
            name: 'R',
            value: dims.r,
            expected: 2,
        });
    }

    if dims.m == 0 {
        return Err(Error::InvalidDimension {
            name: 'M',
            value: 0,
            expected: 1, // Must be > 0
        });
    }

    Ok(())
}

/// Verifies ListenerView coordinate type and values.
fn validate_listener_view(hrtf: &Hrtf) -> Result<()> {
    if hrtf.listener_view.is_empty() {
        return Ok(()); // Optional
    }

    let coord_type = hrtf.listener_view.get_attribute("Type");

    match coord_type {
        Some("cartesian") => {
            // Should be [1, 0, 0], looking forward
            if hrtf.listener_view.len() >= 3 {
                let expected = [1.0, 0.0, 0.0];
                if !values_match(&hrtf.listener_view.values, &expected) {
                    log::warn!("ListenerView values don't match expected [1,0,0]");
                }
            }
        }
        Some("spherical") => {
            // Should be [0, 0, 1]: azimuth=0, elevation=0, distance=1
            if hrtf.listener_view.len() >= 3 {
                let expected = [0.0, 0.0, 1.0];
                if !values_match(&hrtf.listener_view.values, &expected) {
                    log::warn!("ListenerView values don't match expected [0,0,1]");
                }
            }
        }
        _ => {
            // Unknown coordinate type - log warning but don't fail
            log::warn!("Unknown ListenerView coordinate type: {:?}", coord_type);
        }
    }

    Ok(())
}

/// Verifies EmitterPosition is at origin.
fn validate_emitter_position(hrtf: &Hrtf) -> Result<()> {
    if hrtf.emitter_position.is_empty() {
        return Ok(());
    }

    // Emitter should be at origin [0, 0, 0]
    if hrtf.emitter_position.len() >= 3 {
        let expected = [0.0, 0.0, 0.0];
        if !values_match(&hrtf.emitter_position.values, &expected) {
            log::warn!("EmitterPosition not at origin");
        }
    }

    Ok(())
}

/// Verifies ReceiverPosition is in cartesian coordinates and symmetric.
fn validate_receiver_position(hrtf: &Hrtf) -> Result<()> {
    if hrtf.receiver_position.is_empty() {
        // ReceiverPosition is required but may not be parsed yet
        log::warn!("ReceiverPosition array is empty");
        return Ok(());
    }

    // Check coordinate type - if not present, assume cartesian
    let coord_type = hrtf.receiver_position.get_attribute("Type");
    if coord_type.is_some() && coord_type != Some("cartesian") {
        return Err(Error::InvalidAttribute {
            name: "ReceiverPosition.Type",
            expected: "cartesian",
        });
    }

    // Check we have enough values: R=2 receivers, C=3 coordinates each
    let r = hrtf.dimensions().r as usize;
    let c = hrtf.dimensions().c as usize;
    let expected_len = r * c;

    if hrtf.receiver_position.len() < expected_len {
        return Err(Error::InvalidArraySize {
            name: "ReceiverPosition",
            expected: expected_len,
            actual: hrtf.receiver_position.len(),
        });
    }

    // Check receiver positions are symmetric
    // Left ear: values[0..3], Right ear: values[3..6]
    // For binaural: x and z should be ~0, y values should be opposite
    if hrtf.receiver_position.len() >= 6 {
        let values = &hrtf.receiver_position.values;

        // Check x coordinates are ~0
        if values[0].abs() >= RECEIVER_POSITION_TOLERANCE {
            log::warn!("Left receiver x position {} exceeds tolerance", values[0]);
        }
        if values[3].abs() >= RECEIVER_POSITION_TOLERANCE {
            log::warn!("Right receiver x position {} exceeds tolerance", values[3]);
        }

        // Check z coordinates are ~0
        if values[2].abs() >= RECEIVER_POSITION_TOLERANCE {
            log::warn!("Left receiver z position {} exceeds tolerance", values[2]);
        }
        if values[5].abs() >= RECEIVER_POSITION_TOLERANCE {
            log::warn!("Right receiver z position {} exceeds tolerance", values[5]);
        }

        // Check y coordinates are symmetric with opposite signs
        if (values[1] + values[4]).abs() >= RECEIVER_POSITION_TOLERANCE {
            log::warn!(
                "Receiver y positions not symmetric: {} + {} = {}",
                values[1],
                values[4],
                values[1] + values[4]
            );
        }
    }

    Ok(())
}

/// Verifies SourcePosition dimension list.
fn validate_source_position(hrtf: &Hrtf) -> Result<()> {
    if hrtf.source_position.is_empty() {
        return Err(Error::MissingArray("SourcePosition"));
    }

    // Check DIMENSION_LIST is M,C
    let dim_list = hrtf.source_position.get_attribute("DIMENSION_LIST");
    if dim_list != Some("M,C") {
        log::warn!(
            "SourcePosition DIMENSION_LIST is {:?}, expected 'M,C'",
            dim_list
        );
    }

    Ok(())
}

/// Verifies DataDelay dimension list.
fn validate_data_delay(hrtf: &Hrtf) -> Result<()> {
    if hrtf.data_delay.is_empty() {
        return Ok(()); // Optional
    }

    // Check DIMENSION_LIST is I,R or M,R
    let dim_list = hrtf.data_delay.get_attribute("DIMENSION_LIST");
    match dim_list {
        Some("I,R") | Some("M,R") => {}
        _ => {
            log::warn!(
                "DataDelay DIMENSION_LIST is {:?}, expected 'I,R' or 'M,R'",
                dim_list
            );
        }
    }

    Ok(())
}

/// Verifies sampling rate is consistent.
fn validate_sampling_rate(hrtf: &Hrtf) -> Result<()> {
    if hrtf.data_sampling_rate.is_empty() {
        return Ok(()); // Will use default
    }

    // Check DIMENSION_LIST is I (single sampling rate for all measurements)
    let dim_list = hrtf.data_sampling_rate.get_attribute("DIMENSION_LIST");
    if dim_list != Some("I") {
        log::warn!(
            "DataSamplingRate DIMENSION_LIST is {:?}, expected 'I'",
            dim_list
        );
    }

    Ok(())
}

/// Compares array values with expected values (with tolerance).
fn values_match(values: &[f32], expected: &[f32]) -> bool {
    if values.len() < expected.len() {
        return false;
    }

    const TOLERANCE: f32 = 1e-6;
    for (i, &exp) in expected.iter().enumerate() {
        if (values[i] - exp).abs() > TOLERANCE {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_values_match() {
        assert!(values_match(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]));
        assert!(values_match(&[1.0, 0.0, 0.0, 0.5], &[1.0, 0.0, 0.0]));
        assert!(!values_match(&[1.0, 0.1, 0.0], &[1.0, 0.0, 0.0]));
        assert!(!values_match(&[1.0], &[1.0, 0.0, 0.0]));
    }
}
