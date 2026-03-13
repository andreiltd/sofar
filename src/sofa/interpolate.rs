//! Filter interpolation using inverse distance weighting.
//!
//! Combines the nearest measurement with up to 6 directional neighbors
//! using inverse distance weighting for smooth HRTF interpolation.

use arrayvec::ArrayVec;

use super::kdtree::Point3;
use super::neighbors::Neighborhood;
use super::reader::Hrtf;

/// Interpolated HRTF filter result.
#[derive(Debug, Clone)]
pub struct InterpolatedFilter {
    /// Left channel impulse response.
    pub left: Vec<f32>,
    /// Right channel impulse response.
    pub right: Vec<f32>,
    /// Left channel delay in samples.
    pub delay_left: f32,
    /// Right channel delay in samples.
    pub delay_right: f32,
}

/// Interpolate HRTF filter at a given position.
///
/// Uses inverse distance weighting to combine the nearest measurement
/// with up to 6 directional neighbors.
///
/// # Arguments
/// * `hrtf` - The HRTF data
/// * `neighborhood` - Precomputed neighbor relationships
/// * `nearest_idx` - Index of the nearest measurement
/// * `position` - The query position in cartesian coordinates
///
/// # Returns
/// An interpolated filter, or None if the data is invalid.
pub fn interpolate(
    hrtf: &Hrtf,
    neighborhood: &Neighborhood,
    nearest_idx: usize,
    position: &Point3,
) -> Option<InterpolatedFilter> {
    let dims = hrtf.dimensions();
    let n = dims.n as usize;
    let r = dims.r as usize;
    let m = dims.m as usize;
    let c = dims.c as usize;

    if r < 2 || nearest_idx >= m {
        return None;
    }

    let ir_values = &hrtf.data_ir.values;
    let delay_values = &hrtf.data_delay.values;
    let source_pos = &hrtf.source_position.values;

    // Get the nearest position
    let nearest_offset = nearest_idx * c;
    if nearest_offset + 2 >= source_pos.len() {
        return None;
    }
    let nearest_pos: Point3 = [
        source_pos[nearest_offset],
        source_pos[nearest_offset + 1],
        source_pos[nearest_offset + 2],
    ];

    // Calculate distance to nearest
    let nearest_dist = distance(&nearest_pos, position);

    // If very close (exact match), just return the nearest filter
    if nearest_dist < 1e-6 {
        return extract_filter(hrtf, nearest_idx, n, r);
    }

    // Get neighbors
    let neighbors = neighborhood.get(nearest_idx)?;

    // Collect points with their distances and weights. At most 7: nearest + 6 neighbors.
    let mut points: ArrayVec<(usize, f32), 7> = ArrayVec::new();
    points.push((nearest_idx, 1.0 / nearest_dist));

    // Add the closer neighbor from each directional pair
    add_closer_neighbor(
        &mut points,
        neighbors.phi_plus,
        neighbors.phi_minus,
        position,
        source_pos,
        c,
    );
    add_closer_neighbor(
        &mut points,
        neighbors.theta_plus,
        neighbors.theta_minus,
        position,
        source_pos,
        c,
    );
    add_closer_neighbor(
        &mut points,
        neighbors.radius_plus,
        neighbors.radius_minus,
        position,
        source_pos,
        c,
    );

    // Compute weight sum for normalization
    let weight_sum: f32 = points.iter().map(|(_, w)| w).sum();
    if weight_sum < 1e-10 {
        return extract_filter(hrtf, nearest_idx, n, r);
    }

    // Initialize result
    let mut left = vec![0.0f32; n];
    let mut right = vec![0.0f32; n];
    let mut delay_left = 0.0f32;
    let mut delay_right = 0.0f32;

    // Weighted sum of filters
    for (idx, weight) in &points {
        let norm_weight = weight / weight_sum;

        // IR offsets: data_ir is M * R * N
        let ir_offset_left = idx * r * n;
        let ir_offset_right = ir_offset_left + n;

        // Add weighted IR values
        if ir_offset_right + n <= ir_values.len() {
            for i in 0..n {
                left[i] += ir_values[ir_offset_left + i] * norm_weight;
                right[i] += ir_values[ir_offset_right + i] * norm_weight;
            }
        }

        // Add weighted delay values
        // Delay can be:
        // - per-measurement: M values
        // - per-channel: M*R values
        // - global per-receiver: R values, e.g. 2 for stereo. Matches C behavior.
        if delay_values.len() == m {
            // Single delay per measurement
            delay_left += delay_values[*idx] * norm_weight;
            delay_right += delay_values[*idx] * norm_weight;
        } else if delay_values.len() >= m * r {
            // Per-channel delay
            let delay_offset = idx * r;
            if delay_offset + 1 < delay_values.len() {
                delay_left += delay_values[delay_offset] * norm_weight;
                delay_right += delay_values[delay_offset + 1] * norm_weight;
            }
        } else if delay_values.len() >= r {
            // Global delay values, one per receiver channel
            delay_left += delay_values[0] * norm_weight;
            delay_right += delay_values.get(1).copied().unwrap_or(delay_values[0]) * norm_weight;
        }
    }

    Some(InterpolatedFilter {
        left,
        right,
        delay_left,
        delay_right,
    })
}

/// Add the closer neighbor from a directional pair to the points list.
fn add_closer_neighbor(
    points: &mut ArrayVec<(usize, f32), 7>,
    neighbor_a: Option<usize>,
    neighbor_b: Option<usize>,
    position: &Point3,
    source_pos: &[f32],
    c: usize,
) {
    let mut best_idx = None;
    let mut best_dist = f32::MAX;

    for idx in [neighbor_a, neighbor_b].into_iter().flatten() {
        let offset = idx * c;
        if offset + 2 >= source_pos.len() {
            continue;
        }
        let pos: Point3 = [
            source_pos[offset],
            source_pos[offset + 1],
            source_pos[offset + 2],
        ];
        let dist = distance(&pos, position);
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }

    if let Some(idx) = best_idx
        && best_dist > 1e-10
    {
        points.push((idx, 1.0 / best_dist));
    }
}

/// Extract a single filter without interpolation.
fn extract_filter(hrtf: &Hrtf, idx: usize, n: usize, r: usize) -> Option<InterpolatedFilter> {
    let ir_values = &hrtf.data_ir.values;
    let delay_values = &hrtf.data_delay.values;
    let m = hrtf.dimensions().m as usize;

    let ir_offset_left = idx * r * n;
    let ir_offset_right = ir_offset_left + n;

    if ir_offset_right + n > ir_values.len() {
        return None;
    }

    let left = ir_values[ir_offset_left..ir_offset_left + n].to_vec();
    let right = ir_values[ir_offset_right..ir_offset_right + n].to_vec();

    let (delay_left, delay_right) = if delay_values.len() == m {
        (
            delay_values.get(idx).copied().unwrap_or(0.0),
            delay_values.get(idx).copied().unwrap_or(0.0),
        )
    } else if delay_values.len() >= m * r {
        let offset = idx * r;
        (
            delay_values.get(offset).copied().unwrap_or(0.0),
            delay_values.get(offset + 1).copied().unwrap_or(0.0),
        )
    } else if delay_values.len() >= r {
        // Global delay values, one per receiver channel
        (
            delay_values.first().copied().unwrap_or(0.0),
            delay_values
                .get(1)
                .copied()
                .unwrap_or(delay_values.first().copied().unwrap_or(0.0)),
        )
    } else {
        (0.0, 0.0)
    };

    Some(InterpolatedFilter {
        left,
        right,
        delay_left,
        delay_right,
    })
}

/// Calculate euclidean distance between two points.
#[inline]
fn distance(a: &Point3, b: &Point3) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Get filter without interpolation, using nearest neighbor only.
///
/// # Arguments
/// * `hrtf` - The HRTF data
/// * `nearest_idx` - Index of the nearest measurement
///
/// # Returns
/// The filter at the given index, or None if invalid.
pub fn get_filter_nointerp(hrtf: &Hrtf, nearest_idx: usize) -> Option<InterpolatedFilter> {
    let dims = hrtf.dimensions();
    let n = dims.n as usize;
    let r = dims.r as usize;
    extract_filter(hrtf, nearest_idx, n, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        assert!((distance(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((distance(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]) - 5.0).abs() < 1e-6);
    }
}
