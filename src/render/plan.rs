use std::fmt;
use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex};

use super::filter::FilterLayout;
use super::{Error, FilterTransform, PreparedFilter};

const DEFAULT_SAMPLE_RATE: f32 = 48_000.0;
const DEFAULT_PARTITION_LEN: usize = 256;

/// Immutable FFT layout and plans shared by any number of sources.
#[derive(Clone)]
pub struct RendererPlan(
    /// Shared validated layout and forward/inverse FFT plans.
    pub(super) Arc<PlanData>,
);

/// Validated convolution dimensions and reusable forward/inverse FFT plans.
pub(super) struct PlanData {
    /// Base filter dimensions and sample rate used for compatibility.
    pub layout: FilterLayout,
    /// FIR partition count: `ceil(filter_len / partition_len)`.
    pub partitions: usize,
    /// FFT length in samples: twice the partition length.
    pub fft_len: usize,
    /// Complex bins per real FFT, including DC and Nyquist.
    pub spectra_len: usize,
    /// Bins per filter channel or input history: `partitions * spectra_len`.
    pub spectra_data_len: usize,
    /// Inverse-FFT normalization factor, `1.0 / fft_len`.
    pub inv_scale: f32,
    /// Shared real-to-complex FFT plan for input and filter partitions.
    pub rfft: Arc<dyn RealToComplex<f32>>,
    /// Shared complex-to-real FFT plan for output partitions.
    pub ifft: Arc<dyn ComplexToReal<f32>>,
}

impl RendererPlan {
    /// Configure the immutable FFT layout for filters with `filter_len` taps.
    pub const fn builder(filter_len: usize) -> RendererPlanBuilder {
        RendererPlanBuilder {
            layout: FilterLayout {
                sample_rate: DEFAULT_SAMPLE_RATE,
                filter_len,
                partition_len: DEFAULT_PARTITION_LEN,
            },
        }
    }

    /// Sample rate in hertz, used to convert filter delays to samples.
    pub fn sample_rate(&self) -> f32 {
        self.0.layout.sample_rate
    }

    /// FIR tap count per ear that every filter must have.
    pub fn filter_len(&self) -> usize {
        self.0.layout.filter_len
    }

    /// Samples per convolution partition. Audio block lengths must be
    /// multiples of this length.
    pub fn partition_len(&self) -> usize {
        self.0.layout.partition_len
    }

    /// Allocate reusable filter-preparation scratch, sharing this plan's FFT.
    pub fn filter_transform(&self) -> FilterTransform {
        FilterTransform::new(self.clone())
    }

    /// Check that a prepared filter was created for this plan's layout.
    pub(super) fn validate_filter(&self, filter: &PreparedFilter) -> Result<(), Error> {
        self.0.layout.validate(filter.0.layout)
    }
}

impl fmt::Debug for RendererPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RendererPlan")
            .field("sample_rate", &self.sample_rate())
            .field("filter_len", &self.filter_len())
            .field("partition_len", &self.partition_len())
            .finish_non_exhaustive()
    }
}

/// Builder for immutable FFT configuration, without per-source policies.
#[derive(Clone)]
#[must_use]
pub struct RendererPlanBuilder {
    /// Requested FFT configuration, validated when building the plan.
    layout: FilterLayout,
}

impl RendererPlanBuilder {
    /// Set the sample rate. Defaults to 48,000 Hz.
    pub const fn with_sample_rate(mut self, sample_rate: f32) -> Self {
        self.layout.sample_rate = sample_rate;

        self
    }

    /// Set the convolution partition length. Defaults to 256 samples.
    pub const fn with_partition_len(mut self, partition_len: usize) -> Self {
        self.layout.partition_len = partition_len;

        self
    }

    /// Validate buffer sizes and plan the forward and inverse FFTs.
    pub fn build(self) -> Result<RendererPlan, Error> {
        let FilterLayout {
            sample_rate,
            filter_len,
            partition_len,
        } = self.layout;

        if !sample_rate.is_normal() || sample_rate.is_sign_negative() {
            return Err(Error::InvalidSampleRate { sample_rate });
        }

        if partition_len == 0 {
            return Err(Error::InvalidPartitionLength { partition_len });
        }

        if filter_len == 0 {
            return Err(Error::ZeroFilterLength);
        }

        // Bound allocation sizes in bytes before creating buffers or FFT plans.
        let fft_len = partition_len
            .checked_mul(2)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f32>())
            .ok_or(Error::LayoutTooLarge)?;

        let spectra_len = partition_len + 1;
        let partitions = filter_len.div_ceil(partition_len);
        let spectra_data_len = partitions
            .checked_mul(spectra_len)
            .filter(|&len| len <= isize::MAX as usize / size_of::<Complex<f32>>())
            .ok_or(Error::LayoutTooLarge)?;

        let mut planner = RealFftPlanner::new();

        Ok(RendererPlan(Arc::new(PlanData {
            layout: self.layout,
            partitions,
            fft_len,
            spectra_len,
            spectra_data_len,
            inv_scale: 1.0 / fft_len as f32,
            rfft: planner.plan_fft_forward(fft_len),
            ifft: planner.plan_fft_inverse(fft_len),
        })))
    }
}

impl fmt::Debug for RendererPlanBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RendererPlanBuilder")
            .field("sample_rate", &self.layout.sample_rate)
            .field("filter_len", &self.layout.filter_len)
            .field("partition_len", &self.layout.partition_len)
            .finish()
    }
}
