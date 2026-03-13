//! Resampling for HRTF filters.
//!
//! This module provides resampling functionality for HRTF IR data using
//! either the `rubato` crate, behind a feature flag, or a simple linear interpolation
//! fallback.

use super::reader::Hrtf;

#[cfg(feature = "resample")]
use audioadapter_buffers::direct::SequentialSliceOfVecs;
#[cfg(feature = "resample")]
use rubato::audioadapter::Adapter;
#[cfg(feature = "resample")]
use rubato::{Fft, FixedSync, Resampler};

/// Resample HRTF data to a target sample rate.
///
/// This modifies the HRTF data in place, updating:
/// - IR filter coefficients, resampled to the new rate
/// - Filter length N
/// - Data delay values, scaled proportionally
/// - Sample rate
///
/// # Arguments
/// * `hrtf` - The HRTF data to resample
/// * `target_rate` - The desired sample rate in Hz
///
/// # Returns
/// Ok if resampling succeeded, or an error message.
#[cfg(feature = "resample")]
pub fn resample(hrtf: &mut Hrtf, target_rate: f32) -> Result<(), String> {
    let source_rate = hrtf.sample_rate();
    if (source_rate - target_rate).abs() < 0.1 {
        return Ok(()); // Already at target rate
    }

    let ratio = target_rate as f64 / source_rate as f64;
    let dims = hrtf.dimensions();
    let m = dims.m as usize;
    let r = dims.r as usize;
    let n = dims.n as usize;
    let new_n = (n as f64 * ratio).ceil() as usize;

    // Create resampler
    let chunk_size = n;
    let mut resampler = Fft::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        chunk_size,
        1, // One sub-chunk
        1, // One channel at a time
        FixedSync::Input,
    )
    .map_err(|e| format!("Failed to create resampler: {}", e))?;

    let output_delay = resampler.output_delay();

    // Zero-filled chunk used to flush the resampler pipeline
    let flush_data = vec![vec![0.0f32; n]];
    let flush_adapter = SequentialSliceOfVecs::new(&flush_data, 1, n)
        .map_err(|e| format!("Flush adapter error: {}", e))?;

    // Prepare new IR buffer
    let mut new_ir = vec![0.0f32; m * r * new_n];

    // Process each filter
    for measurement in 0..m {
        for channel in 0..r {
            let src_offset = measurement * r * n + channel * n;
            let dst_offset = measurement * r * new_n + channel * new_n;

            // Get source samples as a single-channel buffer
            if src_offset + n > hrtf.data_ir.values.len() {
                continue;
            }
            let input_data = vec![hrtf.data_ir.values[src_offset..src_offset + n].to_vec()];
            let input_adapter = SequentialSliceOfVecs::new(&input_data, 1, n)
                .map_err(|e| format!("Input adapter error: {}", e))?;

            // Reset resampler state
            resampler.reset();

            // The FFT resampler has internal latency (output_delay frames).
            // A single process() call buffers the input without producing output.
            // We must flush with zero-padded chunks and skip the delay.
            let mut all_output: Vec<f32> = Vec::with_capacity(output_delay + new_n);
            let total_needed = output_delay + new_n;

            // Feed actual data
            let output = resampler
                .process(&input_adapter, 0, None)
                .map_err(|e| format!("Resampling failed: {}", e))?;
            collect_frames(&output, &mut all_output);

            // Flush with zeros until we have enough output
            while all_output.len() < total_needed {
                let output = resampler
                    .process(&flush_adapter, 0, None)
                    .map_err(|e| format!("Resampling flush failed: {}", e))?;
                if output.frames() == 0 {
                    break;
                }
                collect_frames(&output, &mut all_output);
            }

            // Skip the delay and copy resampled data
            let start = output_delay.min(all_output.len());
            let copy_len = new_n.min(all_output.len().saturating_sub(start));
            new_ir[dst_offset..dst_offset + copy_len]
                .copy_from_slice(&all_output[start..start + copy_len]);
        }
    }

    // Update IR data
    hrtf.data_ir.values = new_ir;

    // Scale delay values
    for delay in &mut hrtf.data_delay.values {
        *delay *= ratio as f32;
    }

    // Update dimensions
    hrtf.set_n(new_n as u32);
    hrtf.set_sample_rate(target_rate);

    Ok(())
}

#[cfg(feature = "resample")]
fn collect_frames<'a>(output: &impl Adapter<'a, f32>, dest: &mut Vec<f32>) {
    for i in 0..output.frames() {
        dest.push(output.read_sample(0, i).unwrap_or(0.0));
    }
}

/// Simple linear interpolation fallback when rubato is not available.
///
/// This provides basic resampling functionality with lower quality than
/// the rubato-based implementation.
#[cfg(not(feature = "resample"))]
pub fn resample(hrtf: &mut Hrtf, target_rate: f32) -> Result<(), String> {
    let source_rate = hrtf.sample_rate();
    if (source_rate - target_rate).abs() < 0.1 {
        return Ok(()); // Already at target rate
    }

    let ratio = target_rate / source_rate;
    let dims = hrtf.dimensions();
    let m = dims.m as usize;
    let r = dims.r as usize;
    let n = dims.n as usize;
    let new_n = (n as f32 * ratio).ceil() as usize;

    // Prepare new IR buffer
    let mut new_ir = vec![0.0f32; m * r * new_n];

    // Process each filter with linear interpolation
    for measurement in 0..m {
        for channel in 0..r {
            let src_offset = measurement * r * n + channel * n;
            let dst_offset = measurement * r * new_n + channel * new_n;

            if src_offset + n > hrtf.data_ir.values.len() {
                continue;
            }

            let src = &hrtf.data_ir.values[src_offset..src_offset + n];

            for i in 0..new_n {
                let src_pos = i as f32 / ratio;
                let idx = src_pos as usize;
                let frac = src_pos - idx as f32;

                let sample = if idx + 1 < n {
                    src[idx] * (1.0 - frac) + src[idx + 1] * frac
                } else if idx < n {
                    src[idx]
                } else {
                    0.0
                };

                new_ir[dst_offset + i] = sample;
            }
        }
    }

    // Update IR data
    hrtf.data_ir.values = new_ir;

    // Scale delay values
    for delay in &mut hrtf.data_delay.values {
        *delay *= ratio;
    }

    // Update dimensions
    hrtf.set_n(new_n as u32);
    hrtf.set_sample_rate(target_rate);

    Ok(())
}
