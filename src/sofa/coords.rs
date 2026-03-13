//! Coordinate conversion utilities for SOFA/HRTF data.
//!
//! SOFA files can store positions in either cartesian [x, y, z] or
//! spherical [azimuth, elevation, radius] coordinates. This module
//! provides functions to convert between these representations.
//!
//! ## Coordinate Systems
//!
//! **Cartesian: x, y, z** in meters:
//! - x: forward direction
//! - y: left direction  
//! - z: up direction
//!
//! **Spherical: azimuth, elevation, radius**:
//! - azimuth: angle in degrees, 0° = front, 90° = left, 180°/-180° = back
//! - elevation: angle in degrees, 0° = horizontal, 90° = up, -90° = down
//! - radius: distance in meters

use std::f32::consts::PI;

/// Convert cartesian coordinates to spherical.
///
/// Input: [x, y, z] in meters.
/// Output: [azimuth, elevation, radius] where azimuth/elevation are in degrees.
pub fn cartesian_to_spherical(cartesian: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = cartesian;

    let r = (x * x + y * y + z * z).sqrt();
    let theta = z.atan2((x * x + y * y).sqrt()); // elevation
    let phi = y.atan2(x); // azimuth

    [
        (phi * (180.0 / PI) + 360.0) % 360.0, // azimuth in degrees, 0-360
        theta * (180.0 / PI),                 // elevation in degrees
        r,                                    // radius in meters
    ]
}

/// Convert spherical coordinates to cartesian.
///
/// Input: [azimuth, elevation, radius] where azimuth/elevation are in degrees.
/// Output: [x, y, z] in meters.
pub fn spherical_to_cartesian(spherical: [f32; 3]) -> [f32; 3] {
    let [azimuth, elevation, radius] = spherical;

    let phi = azimuth * (PI / 180.0);
    let theta = elevation * (PI / 180.0);

    let horizontal_dist = theta.cos() * radius;

    [
        phi.cos() * horizontal_dist, // x
        phi.sin() * horizontal_dist, // y
        theta.sin() * radius,        // z
    ]
}

/// Compute the distance from origin of a cartesian point.
pub fn radius(cartesian: &[f32; 3]) -> f32 {
    (cartesian[0].powi(2) + cartesian[1].powi(2) + cartesian[2].powi(2)).sqrt()
}

/// Convert an array of cartesian coordinates to spherical in-place.
///
/// The array should contain triplets of [x, y, z] values.
/// After conversion, each triplet becomes [azimuth, elevation, radius].
#[allow(dead_code)]
pub fn convert_array_to_spherical(values: &mut [f32]) {
    for chunk in values.chunks_exact_mut(3) {
        let cart: [f32; 3] = [chunk[0], chunk[1], chunk[2]];
        let sph = cartesian_to_spherical(cart);
        chunk[0] = sph[0];
        chunk[1] = sph[1];
        chunk[2] = sph[2];
    }
}

/// Convert an array of spherical coordinates to cartesian in-place.
///
/// The array should contain triplets of [azimuth, elevation, radius] values.
/// After conversion, each triplet becomes [x, y, z].
pub fn convert_array_to_cartesian(values: &mut [f32]) {
    for chunk in values.chunks_exact_mut(3) {
        let sph: [f32; 3] = [chunk[0], chunk[1], chunk[2]];
        let cart = spherical_to_cartesian(sph);
        chunk[0] = cart[0];
        chunk[1] = cart[1];
        chunk[2] = cart[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_cartesian_to_spherical_front() {
        // Front: [1, 0, 0] -> [0°, 0°, 1m]
        let cart = [1.0, 0.0, 0.0];
        let sph = cartesian_to_spherical(cart);
        assert!(approx_eq(sph[0], 0.0), "azimuth: {} != 0", sph[0]);
        assert!(approx_eq(sph[1], 0.0), "elevation: {} != 0", sph[1]);
        assert!(approx_eq(sph[2], 1.0), "radius: {} != 1", sph[2]);
    }

    #[test]
    fn test_cartesian_to_spherical_left() {
        // Left: [0, 1, 0] -> [90°, 0°, 1m]
        let cart = [0.0, 1.0, 0.0];
        let sph = cartesian_to_spherical(cart);
        assert!(approx_eq(sph[0], 90.0), "azimuth: {} != 90", sph[0]);
        assert!(approx_eq(sph[1], 0.0), "elevation: {} != 0", sph[1]);
        assert!(approx_eq(sph[2], 1.0), "radius: {} != 1", sph[2]);
    }

    #[test]
    fn test_cartesian_to_spherical_up() {
        // Up: [0, 0, 1] -> [0°, 90°, 1m]
        let cart = [0.0, 0.0, 1.0];
        let sph = cartesian_to_spherical(cart);
        assert!(approx_eq(sph[1], 90.0), "elevation: {} != 90", sph[1]);
        assert!(approx_eq(sph[2], 1.0), "radius: {} != 1", sph[2]);
    }

    #[test]
    fn test_spherical_to_cartesian_front() {
        // [0°, 0°, 1m] -> [1, 0, 0]
        let sph = [0.0, 0.0, 1.0];
        let cart = spherical_to_cartesian(sph);
        assert!(approx_eq(cart[0], 1.0), "x: {} != 1", cart[0]);
        assert!(approx_eq(cart[1], 0.0), "y: {} != 0", cart[1]);
        assert!(approx_eq(cart[2], 0.0), "z: {} != 0", cart[2]);
    }

    #[test]
    fn test_spherical_to_cartesian_left() {
        // [90°, 0°, 1m] -> [0, 1, 0]
        let sph = [90.0, 0.0, 1.0];
        let cart = spherical_to_cartesian(sph);
        assert!(approx_eq(cart[0], 0.0), "x: {} != 0", cart[0]);
        assert!(approx_eq(cart[1], 1.0), "y: {} != 1", cart[1]);
        assert!(approx_eq(cart[2], 0.0), "z: {} != 0", cart[2]);
    }

    #[test]
    fn test_roundtrip_cartesian_spherical() {
        let original = [0.5, 0.3, 0.7];
        let spherical = cartesian_to_spherical(original);
        let back = spherical_to_cartesian(spherical);

        assert!(
            approx_eq(original[0], back[0]),
            "x: {} != {}",
            original[0],
            back[0]
        );
        assert!(
            approx_eq(original[1], back[1]),
            "y: {} != {}",
            original[1],
            back[1]
        );
        assert!(
            approx_eq(original[2], back[2]),
            "z: {} != {}",
            original[2],
            back[2]
        );
    }

    #[test]
    fn test_convert_array_to_spherical() {
        let mut values = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // front, left
        convert_array_to_spherical(&mut values);

        // First point: front
        assert!(approx_eq(values[0], 0.0)); // azimuth
        assert!(approx_eq(values[1], 0.0)); // elevation
        assert!(approx_eq(values[2], 1.0)); // radius

        // Second point: left
        assert!(approx_eq(values[3], 90.0)); // azimuth
        assert!(approx_eq(values[4], 0.0)); // elevation
        assert!(approx_eq(values[5], 1.0)); // radius
    }

    #[test]
    fn test_radius() {
        assert!(approx_eq(radius(&[1.0, 0.0, 0.0]), 1.0));
        assert!(approx_eq(radius(&[0.0, 1.0, 0.0]), 1.0));
        assert!(approx_eq(radius(&[0.0, 0.0, 1.0]), 1.0));
        assert!(approx_eq(radius(&[1.0, 1.0, 1.0]), 3.0_f32.sqrt()));
    }
}
