//! Neighbor computation for HRTF interpolation.
//!
//! Computes directional neighbors (phi+/-, theta+/-, radius+/-) for each
//! measurement position to enable smooth interpolation.

use super::coords::{cartesian_to_spherical, spherical_to_cartesian};
use super::kdtree::Point3;
use super::lookup::Lookup;
use super::reader::Hrtf;

/// Maximum angle to search for neighbors (in degrees).
const MAX_SEARCH_ANGLE: f32 = 45.0;

/// Neighbor indices for one measurement position.
///
/// Contains indices to neighboring measurements in 6 directions:
/// phi+/-, theta+/-, radius+/-. A value of `None` means no neighbor
/// was found in that direction.
#[derive(Debug, Clone, Default)]
pub struct Neighbors {
    /// Neighbor in positive phi, i.e. azimuth, direction.
    pub phi_plus: Option<usize>,
    /// Neighbor in negative phi, i.e. azimuth, direction.
    pub phi_minus: Option<usize>,
    /// Neighbor in positive theta, i.e. elevation, direction.
    pub theta_plus: Option<usize>,
    /// Neighbor in negative theta, i.e. elevation, direction.
    pub theta_minus: Option<usize>,
    /// Neighbor in positive radius direction.
    pub radius_plus: Option<usize>,
    /// Neighbor in negative radius direction.
    pub radius_minus: Option<usize>,
}

/// Neighborhood structure containing precomputed neighbors for all positions.
#[derive(Debug)]
pub struct Neighborhood {
    /// Neighbors for each measurement position (length = M).
    neighbors: Vec<Neighbors>,
    /// Step size for angular search (degrees).
    #[allow(dead_code)]
    angle_step: f32,
    /// Step size for radius search (meters).
    #[allow(dead_code)]
    radius_step: f32,
}

impl Neighborhood {
    /// Build neighborhood structure from HRTF data.
    ///
    /// # Arguments
    /// * `hrtf` - The HRTF data
    /// * `lookup` - The spatial lookup structure
    /// * `angle_step` - Step size for angular search in degrees (default: 0.5)
    /// * `radius_step` - Step size for radius search in meters (default: 0.01)
    pub fn new(hrtf: &Hrtf, lookup: &Lookup, angle_step: f32, radius_step: f32) -> Self {
        let m = hrtf.dimensions().m as usize;
        let c = hrtf.dimensions().c as usize;
        let source_pos = &hrtf.source_position.values;

        let mut neighbors = Vec::with_capacity(m);

        for i in 0..m {
            let offset = i * c;
            if offset + 2 >= source_pos.len() {
                neighbors.push(Neighbors::default());
                continue;
            }

            let pos: Point3 = [
                source_pos[offset],
                source_pos[offset + 1],
                source_pos[offset + 2],
            ];

            neighbors.push(Self::find_neighbors(
                i,
                &pos,
                lookup,
                angle_step,
                radius_step,
            ));
        }

        Self {
            neighbors,
            angle_step,
            radius_step,
        }
    }

    /// Find all directional neighbors for a position.
    fn find_neighbors(
        current_idx: usize,
        pos: &Point3,
        lookup: &Lookup,
        angle_step: f32,
        radius_step: f32,
    ) -> Neighbors {
        let spherical = cartesian_to_spherical(*pos);
        let phi = spherical[0];
        let theta = spherical[1];
        let r = spherical[2];

        Neighbors {
            phi_plus: Self::search_direction(
                current_idx,
                phi,
                theta,
                r,
                angle_step,
                0.0,
                0.0,
                lookup,
            ),
            phi_minus: Self::search_direction(
                current_idx,
                phi,
                theta,
                r,
                -angle_step,
                0.0,
                0.0,
                lookup,
            ),
            theta_plus: Self::search_direction(
                current_idx,
                phi,
                theta,
                r,
                0.0,
                angle_step,
                0.0,
                lookup,
            ),
            theta_minus: Self::search_direction(
                current_idx,
                phi,
                theta,
                r,
                0.0,
                -angle_step,
                0.0,
                lookup,
            ),
            radius_plus: Self::search_direction(
                current_idx,
                phi,
                theta,
                r,
                0.0,
                0.0,
                radius_step,
                lookup,
            ),
            radius_minus: Self::search_direction(
                current_idx,
                phi,
                theta,
                r,
                0.0,
                0.0,
                -radius_step,
                lookup,
            ),
        }
    }

    /// Search in a direction until finding a different measurement point.
    #[allow(clippy::too_many_arguments)]
    fn search_direction(
        current_idx: usize,
        phi: f32,
        theta: f32,
        radius: f32,
        phi_step: f32,
        theta_step: f32,
        radius_step: f32,
        lookup: &Lookup,
    ) -> Option<usize> {
        let max_steps = if phi_step.abs() > 0.0 || theta_step.abs() > 0.0 {
            (MAX_SEARCH_ANGLE / phi_step.abs().max(theta_step.abs())).ceil() as i32
        } else {
            // For radius, search until bounds
            100
        };

        for step in 1..=max_steps {
            let step_f = step as f32;
            let new_phi = phi + phi_step * step_f;
            let new_theta = (theta + theta_step * step_f).clamp(-90.0, 90.0);
            let new_radius = (radius + radius_step * step_f).max(0.001);

            // Skip if radius would be outside bounds (matching C behavior: ± radius_step)
            if new_radius < lookup.radius_min - radius_step.abs()
                || new_radius > lookup.radius_max + radius_step.abs()
            {
                break;
            }

            let cart = spherical_to_cartesian([new_phi, new_theta, new_radius]);

            if let Some(idx) = lookup.find(&cart)
                && idx != current_idx
            {
                return Some(idx);
            }
        }

        None
    }

    /// Get neighbors for a measurement index.
    pub fn get(&self, index: usize) -> Option<&Neighbors> {
        self.neighbors.get(index)
    }

    /// Get the angle step used for neighbor computation.
    #[allow(dead_code)]
    pub fn angle_step(&self) -> f32 {
        self.angle_step
    }

    /// Get the radius step used for neighbor computation.
    #[allow(dead_code)]
    pub fn radius_step(&self) -> f32 {
        self.radius_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbors_default() {
        let n = Neighbors::default();
        assert!(n.phi_plus.is_none());
        assert!(n.phi_minus.is_none());
        assert!(n.theta_plus.is_none());
        assert!(n.theta_minus.is_none());
        assert!(n.radius_plus.is_none());
        assert!(n.radius_minus.is_none());
    }
}
