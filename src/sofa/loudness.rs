//! Loudness normalization for HRTF data.
//!
//! Normalizes all HRTF filters so that the frontal filter has unit energy,
//! ensuring consistent perceived loudness across all directions.

use super::coords::cartesian_to_spherical;
use super::reader::Hrtf;

/// Compute the energy, i.e. the sum of squared samples, of a signal.
///
/// # Arguments
/// * `samples` - The audio samples
///
/// # Returns
/// The total energy, computed as the sum of squares.
pub fn loudness(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s * s).sum()
}

/// Find the index of the frontal filter.
///
/// The frontal filter is the one closest to 0° azimuth and 0° elevation.
/// Among ties, prefers the one with maximum radius.
///
/// # Arguments
/// * `hrtf` - The HRTF data
///
/// # Returns
/// The index of the frontal filter, or None if source positions are empty.
pub fn find_frontal_index(hrtf: &Hrtf) -> Option<usize> {
    let source_pos = &hrtf.source_position.values;
    let c = hrtf.dimensions().c as usize;
    let m = hrtf.dimensions().m as usize;

    if c != 3 || source_pos.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_sum = f32::MAX;
    let mut best_radius = f32::MIN;

    for i in 0..m {
        let offset = i * c;
        if offset + 2 >= source_pos.len() {
            break;
        }

        let pos = [
            source_pos[offset],
            source_pos[offset + 1],
            source_pos[offset + 2],
        ];

        // Convert to spherical: [azimuth, elevation, radius]
        let spherical = cartesian_to_spherical(pos);
        let azimuth = spherical[0];
        let elevation = spherical[1];
        let radius = spherical[2];

        // Normalize azimuth to -180..180 for frontal comparison
        let azimuth_norm = if azimuth > 180.0 {
            azimuth - 360.0
        } else {
            azimuth
        };

        // Sum of absolute angles - lower is more frontal
        let sum = azimuth_norm.abs() + elevation.abs();

        // Prefer lower sum, or same sum with larger radius
        if sum < best_sum || (sum == best_sum && radius > best_radius) {
            best_sum = sum;
            best_radius = radius;
            best_idx = i;
        }
    }

    Some(best_idx)
}

/// Normalize HRTF filters for consistent loudness.
///
/// Scales all IR data so that the frontal filter has unit energy.
/// This modifies the HRTF data in place.
///
/// # Arguments
/// * `hrtf` - The HRTF data to normalize
///
/// # Returns
/// The scaling factor applied, or None if normalization failed.
pub fn normalize(hrtf: &mut Hrtf) -> Option<f32> {
    let frontal_idx = find_frontal_index(hrtf)?;

    let n = hrtf.dimensions().n as usize;
    let r = hrtf.dimensions().r as usize;

    // Get the frontal filter's IR data
    let ir_offset = frontal_idx * r * n;
    let ir_end = ir_offset + r * n;

    if ir_end > hrtf.data_ir.values.len() {
        return None;
    }

    // Compute loudness of frontal filter across both channels
    let frontal_energy = loudness(&hrtf.data_ir.values[ir_offset..ir_end]);

    if frontal_energy < 1e-10 {
        return None;
    }

    // Compute scaling factor: sqrt(2 / energy) for unit energy
    let factor = (2.0 / frontal_energy).sqrt();

    // Skip if already normalized (factor ≈ 1)
    if (factor - 1.0).abs() < 1e-6 {
        return Some(1.0);
    }

    // Scale all IR data
    for sample in &mut hrtf.data_ir.values {
        *sample *= factor;
    }

    Some(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness() {
        // Energy of [1, 1, 1, 1] = 4
        assert!((loudness(&[1.0, 1.0, 1.0, 1.0]) - 4.0).abs() < 1e-6);

        // Energy of [0.5, 0.5, 0.5, 0.5] = 1
        assert!((loudness(&[0.5, 0.5, 0.5, 0.5]) - 1.0).abs() < 1e-6);

        // Energy of [3, 4] = 9 + 16 = 25
        assert!((loudness(&[3.0, 4.0]) - 25.0).abs() < 1e-6);

        // Empty slice has zero energy
        assert_eq!(loudness(&[]), 0.0);
    }
}
