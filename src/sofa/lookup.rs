//! Spatial lookup for finding nearest HRTF filters.
//!
//! Provides efficient spatial search for finding the closest HRTF filter
//! to a given 3D position using a KD-tree.

use super::coords::{cartesian_to_spherical, radius};
use super::kdtree::{KdTree, Point3};
use super::reader::Hrtf;

/// Spatial lookup structure for finding nearest HRTF filters.
///
/// Uses a KD-tree for efficient O(log n) nearest neighbor queries.
#[derive(Debug)]
pub struct Lookup {
    /// The KD-tree containing source positions.
    kdtree: KdTree,
    /// Minimum azimuth, phi, in degrees.
    pub phi_min: f32,
    /// Maximum azimuth, phi, in degrees.
    pub phi_max: f32,
    /// Minimum elevation, theta, in degrees.
    pub theta_min: f32,
    /// Maximum elevation, theta, in degrees.
    pub theta_max: f32,
    /// Minimum radius in meters.
    pub radius_min: f32,
    /// Maximum radius in meters.
    pub radius_max: f32,
}

impl Lookup {
    /// Initialize a lookup structure from HRTF data.
    ///
    /// Builds a KD-tree from the source positions and computes
    /// the bounding box in spherical coordinates.
    ///
    /// # Arguments
    /// * `hrtf` - The HRTF data containing source positions
    ///
    /// # Returns
    /// A new Lookup structure, or None if source positions are empty
    /// or not in cartesian coordinates.
    pub fn new(hrtf: &Hrtf) -> Option<Self> {
        let source_pos = &hrtf.source_position.values;
        let c = hrtf.dimensions().c as usize;
        let m = hrtf.dimensions().m as usize;

        if source_pos.is_empty() || c != 3 {
            return None;
        }

        // Build KD-tree and compute spherical bounds
        let mut kdtree = KdTree::new();
        let mut phi_min = f32::MAX;
        let mut phi_max = f32::MIN;
        let mut theta_min = f32::MAX;
        let mut theta_max = f32::MIN;
        let mut radius_min = f32::MAX;
        let mut radius_max = f32::MIN;

        for i in 0..m {
            let offset = i * c;
            if offset + 2 >= source_pos.len() {
                break;
            }

            let pos: Point3 = [
                source_pos[offset],
                source_pos[offset + 1],
                source_pos[offset + 2],
            ];

            // Insert into KD-tree
            kdtree.insert(pos, i);

            // Convert to spherical for bounds
            let spherical = cartesian_to_spherical(pos);
            let phi = spherical[0];
            let theta = spherical[1];
            let r = spherical[2];

            phi_min = phi_min.min(phi);
            phi_max = phi_max.max(phi);
            theta_min = theta_min.min(theta);
            theta_max = theta_max.max(theta);
            radius_min = radius_min.min(r);
            radius_max = radius_max.max(r);
        }

        if kdtree.is_empty() {
            return None;
        }

        Some(Self {
            kdtree,
            phi_min,
            phi_max,
            theta_min,
            theta_max,
            radius_min,
            radius_max,
        })
    }

    /// Find the nearest filter index for a given cartesian coordinate.
    ///
    /// The coordinate may be normalized to fit within the radius bounds
    /// of the available measurements.
    ///
    /// # Arguments
    /// * `coordinate` - The cartesian position [x, y, z] to look up
    ///
    /// # Returns
    /// The index of the nearest filter, or None if lookup fails.
    pub fn find(&self, coordinate: &Point3) -> Option<usize> {
        // Normalize radius if outside bounds
        let pos = self.normalize_radius(*coordinate);
        self.kdtree.nearest(&pos)
    }

    /// Find the nearest filter index, modifying the coordinate in place
    /// to reflect any radius normalization.
    ///
    /// # Arguments
    /// * `coordinate` - The cartesian position [x, y, z] to look up, which may be modified
    ///
    /// # Returns
    /// The index of the nearest filter, or None if lookup fails.
    pub fn find_mut(&self, coordinate: &mut Point3) -> Option<usize> {
        *coordinate = self.normalize_radius(*coordinate);
        self.kdtree.nearest(coordinate)
    }

    /// Normalize a coordinate's radius to fit within bounds.
    fn normalize_radius(&self, mut pos: Point3) -> Point3 {
        let r = radius(&pos);

        if r > self.radius_max {
            let scale = self.radius_max / r;
            pos[0] *= scale;
            pos[1] *= scale;
            pos[2] *= scale;
        } else if r < self.radius_min && r > 0.0 {
            let scale = self.radius_min / r;
            pos[0] *= scale;
            pos[1] *= scale;
            pos[2] *= scale;
        }

        pos
    }

    /// Get the number of measurements in the lookup.
    pub fn is_empty(&self) -> bool {
        self.kdtree.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_radius() {
        let lookup = Lookup {
            kdtree: KdTree::new(),
            phi_min: 0.0,
            phi_max: 360.0,
            theta_min: -90.0,
            theta_max: 90.0,
            radius_min: 1.0,
            radius_max: 2.0,
        };

        // Point at radius 3 should be scaled to radius 2
        let pos = lookup.normalize_radius([3.0, 0.0, 0.0]);
        let r = radius(&pos);
        assert!((r - 2.0).abs() < 1e-5, "radius {} != 2.0", r);

        // Point at radius 0.5 should be scaled to radius 1
        let pos = lookup.normalize_radius([0.5, 0.0, 0.0]);
        let r = radius(&pos);
        assert!((r - 1.0).abs() < 1e-5, "radius {} != 1.0", r);

        // Point at radius 1.5 should remain unchanged
        let pos = lookup.normalize_radius([1.5, 0.0, 0.0]);
        let r = radius(&pos);
        assert!((r - 1.5).abs() < 1e-5, "radius {} != 1.5", r);
    }
}
