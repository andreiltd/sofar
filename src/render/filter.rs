use std::fmt;
use std::sync::Arc;

use realfft::num_complex::Complex;

use crate::filter::Filter;

use super::{Error, RendererPlan};

/// Reason a prepared filter cannot be used with a renderer's FFT layout.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PreparedFilterMismatch {
    /// The filter has a different tap count per ear.
    #[error("filter length differs (filter: {filter}, renderer: {renderer})")]
    FilterLength {
        /// Prepared filter's tap count per ear.
        filter: usize,
        /// Renderer's required tap count per ear.
        renderer: usize,
    },
    /// The filter was partitioned for a different partition length.
    #[error("partition length differs (filter: {filter}, renderer: {renderer})")]
    PartitionLength {
        /// Partition length used for preparation, in samples.
        filter: usize,
        /// Renderer's partition length, in samples.
        renderer: usize,
    },
    /// The filter's delays were converted at a different sample rate.
    #[error("sample rate differs (filter: {filter}, renderer: {renderer})")]
    SampleRate {
        /// Preparation sample rate, in hertz.
        filter: f32,
        /// Renderer's sample rate, in hertz.
        renderer: f32,
    },
}

/// Filter dimensions and sample rate that must match the renderer's layout.
#[derive(Clone, Copy, Debug)]
pub(super) struct FilterLayout {
    /// Sample rate in hertz, also used to convert filter delays.
    pub sample_rate: f32,
    /// FIR tap count per ear before partitioning.
    pub filter_len: usize,
    /// Samples processed per convolution partition.
    pub partition_len: usize,
}

impl FilterLayout {
    /// Check that a filter prepared with layout `other` fits this renderer layout.
    pub fn validate(self, other: Self) -> Result<(), Error> {
        let mismatch = if self.filter_len != other.filter_len {
            PreparedFilterMismatch::FilterLength {
                filter: other.filter_len,
                renderer: self.filter_len,
            }
        } else if self.partition_len != other.partition_len {
            PreparedFilterMismatch::PartitionLength {
                filter: other.partition_len,
                renderer: self.partition_len,
            }
        } else if self.sample_rate.to_bits() != other.sample_rate.to_bits() {
            PreparedFilterMismatch::SampleRate {
                filter: other.sample_rate,
                renderer: self.sample_rate,
            }
        } else {
            return Ok(());
        };

        Err(Error::IncompatiblePreparedFilter { mismatch })
    }
}

/// An immutable stereo filter prepared for a particular FFT layout.
///
/// Cloning shares the spectra and is inexpensive. Destroy the last owner on a
/// non-real-time thread. The same filter can be used by independently created
/// plans with matching sample rate, filter length, and partition length.
#[derive(Clone)]
pub struct PreparedFilter(
    /// Shared owner of both ears' immutable spectra and delay metadata.
    pub(super) Arc<PreparedFilterData>,
);

/// Immutable stereo spectra, layout, and delays shared by prepared-filter handles.
pub(super) struct PreparedFilterData {
    /// Preparation layout required by compatible renderers.
    pub layout: FilterLayout,
    /// Left/right spectra, each flattened as partition-major FFT bins.
    pub spectra: [Box<[Complex<f32>]>; 2],
    /// Left/right delays in whole samples, converted from seconds.
    pub delays: [usize; 2],
}

impl PreparedFilter {
    /// Original FIR length in samples.
    pub fn filter_len(&self) -> usize {
        self.0.layout.filter_len
    }

    /// Partition length used to prepare the filter.
    pub fn partition_len(&self) -> usize {
        self.0.layout.partition_len
    }

    /// Sample rate used to convert delay values to samples.
    pub fn sample_rate(&self) -> f32 {
        self.0.layout.sample_rate
    }

    /// Left and right delays in samples, rounded toward zero.
    pub fn delay_samples(&self) -> [usize; 2] {
        self.0.delays
    }
}

impl fmt::Debug for PreparedFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedFilter")
            .field("filter_len", &self.filter_len())
            .field("partition_len", &self.partition_len())
            .field("sample_rate", &self.sample_rate())
            .field("delay_samples", &self.delay_samples())
            .finish_non_exhaustive()
    }
}

/// Reusable FFT workspace for preparing filters outside the audio callback.
///
/// Preparation reuses scratch buffers but allocates the returned filter's
/// immutable stereo spectra. Construction and cloning also allocate scratch.
/// [`FilterPublisher::publish_filter`](super::FilterPublisher::publish_filter)
/// prepares filters with its own transform.
#[derive(Clone)]
pub struct FilterTransform {
    /// Shared layout and FFT plans used for preparation.
    pub(super) plan: RendererPlan,
    /// Workspace for one zero-padded time-domain partition (`fft_len` samples).
    input: Box<[f32]>,
    /// Reusable forward-FFT workspace sized for the plan.
    scratch: Vec<Complex<f32>>,
}

impl FilterTransform {
    pub(super) fn new(plan: RendererPlan) -> Self {
        Self {
            input: vec![0.0; plan.0.fft_len].into_boxed_slice(),
            scratch: plan.0.rfft.make_scratch_vec(),
            plan,
        }
    }

    /// Transform a time-domain filter, validating both FIR lengths and delays.
    ///
    /// Delays must be finite, nonnegative, and representable in samples. Their
    /// maximum supported value is a renderer option, not a preparation limit.
    pub fn prepare(&mut self, filter: &Filter) -> Result<PreparedFilter, Error> {
        let delays = self.plan.filter_delays(filter)?;

        // Allocate stereo spectra only after both channels and delays are valid.
        let mut prepared = self.plan.allocate_filter();

        self.plan.transform_filter(
            filter,
            delays,
            &mut prepared,
            &mut self.input,
            &mut self.scratch,
        )?;

        Ok(prepared)
    }
}

impl fmt::Debug for FilterTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterTransform")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl RendererPlan {
    /// Check both FIR lengths, then convert both delays to whole samples.
    pub(super) fn filter_delays(&self, filter: &Filter) -> Result<[usize; 2], Error> {
        for taps in [&filter.left, &filter.right] {
            if taps.len() != self.filter_len() {
                return Err(Error::InvalidFilterLength {
                    actual: taps.len(),
                    expected: self.filter_len(),
                });
            }
        }

        Ok([
            delay_samples(filter.ldelay, self.sample_rate())?,
            delay_samples(filter.rdelay, self.sample_rate())?,
        ])
    }

    /// Allocate silent, unshared filter storage for this plan's layout.
    pub(super) fn allocate_filter(&self) -> PreparedFilter {
        let spectra = std::array::from_fn(|_| {
            vec![Complex::new(0.0, 0.0); self.0.spectra_data_len].into_boxed_slice()
        });

        PreparedFilter(Arc::new(PreparedFilterData {
            layout: self.0.layout,
            spectra,
            delays: [0; 2],
        }))
    }

    /// Overwrite unshared filter storage with the spectra of `filter`.
    ///
    /// `delays` must come from [`filter_delays`](Self::filter_delays) for the
    /// same filter, and `output` must have this plan's layout. `input` and
    /// `scratch` are forward-FFT workspaces sized for this plan.
    ///
    /// 1. Record the validated delays in `output`.
    /// 2. For each ear, copy every FIR partition into the zero-padded FFT
    ///    input and transform it into that partition's spectrum. The forward
    ///    FFT writes every bin, so stale spectra need no clearing.
    ///
    /// # Panics
    ///
    /// Panics if `output` is shared with another handle.
    pub(super) fn transform_filter(
        &self,
        filter: &Filter,
        delays: [usize; 2],
        output: &mut PreparedFilter,
        input: &mut [f32],
        scratch: &mut [Complex<f32>],
    ) -> Result<(), Error> {
        let data = Arc::get_mut(&mut output.0).expect("filter storage is unshared");

        debug_assert!(
            data.spectra
                .iter()
                .all(|s| s.len() == self.0.spectra_data_len)
        );

        data.layout = self.0.layout;
        data.delays = delays;

        for (taps, spectra) in [&filter.left, &filter.right]
            .into_iter()
            .zip(&mut data.spectra)
        {
            for (partition, spectrum) in taps
                .chunks(self.partition_len())
                .zip(spectra.chunks_exact_mut(self.0.spectra_len))
            {
                input[..partition.len()].copy_from_slice(partition);

                // Zero the FFT tail, including any unused taps in the last partition.
                input[partition.len()..].fill(0.0);
                self.0.rfft.process_with_scratch(input, spectrum, scratch)?;
            }
        }

        Ok(())
    }
}

/// Convert a filter delay to whole samples, rounding toward zero.
fn delay_samples(delay_seconds: f32, sample_rate: f32) -> Result<usize, Error> {
    let samples = delay_seconds * sample_rate;

    if delay_seconds < 0.0 || !samples.is_finite() || samples >= usize::MAX as f32 {
        return Err(Error::InvalidDelay { delay_seconds });
    }

    Ok(samples as usize)
}

/// Convert a maximum delay to whole samples, rounding up.
///
/// Rounding up keeps every filter delay that is at most `delay_seconds` within
/// the capacity, because [`delay_samples`] rounds toward zero.
pub(super) fn delay_capacity(delay_seconds: f32, sample_rate: f32) -> Result<usize, Error> {
    let samples = (delay_seconds * sample_rate).ceil();

    if delay_seconds < 0.0 || !samples.is_finite() || samples >= usize::MAX as f32 {
        return Err(Error::InvalidDelay { delay_seconds });
    }

    Ok(samples as usize)
}
