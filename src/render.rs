//! Uniformly partitioned convolution for HRTF rendering.
//!
//! [`RendererPlan`] shares immutable FFT plans. Each [`Renderer`] owns its own
//! input history, delay lines, and transition state. To update filters from a
//! worker thread while an audio callback renders, convert a renderer with
//! [`Renderer::into_realtime`].
//!
//! See Chapter 5 of [Partitioned convolution algorithms for real-time
//! auralization](https://publications.rwth-aachen.de/record/466561/files/466561.pdf).

use std::fmt;

use crate::filter::Filter;

mod builder;
mod convolution;
mod filter;
mod plan;
mod realtime;

pub use builder::RendererBuilder;
pub use filter::{FilterTransform, PreparedFilter, PreparedFilterMismatch};
pub use plan::{RendererPlan, RendererPlanBuilder};
pub use realtime::{FilterPublisher, RealtimeRenderer};

use convolution::{Engine, StereoBlock, stereo_block};

/// Failures in renderer configuration, filter updates, or audio processing.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Filter length must be greater than zero")]
    ZeroFilterLength,
    #[error("Renderer layout exceeds the supported buffer size")]
    LayoutTooLarge,
    #[error("Crossfade duration must be greater than zero")]
    InvalidCrossfadeDuration,
    #[error("The real-time renderer has been dropped")]
    RendererDisconnected,
    #[error("Sample rate is invalid: {sample_rate}")]
    InvalidSampleRate { sample_rate: f32 },
    #[error("Partition length is invalid: {partition_len}")]
    InvalidPartitionLength { partition_len: usize },
    #[error("Filter length mismatch: expected {expected}, got {actual}")]
    InvalidFilterLength { actual: usize, expected: usize },
    #[error("Delay is invalid: {delay_seconds} seconds")]
    InvalidDelay { delay_seconds: f32 },
    #[error("Delay capacity is too large: {max_samples} samples")]
    DelayCapacityTooLarge { max_samples: usize },
    #[error("Prepared filter is incompatible with renderer: {mismatch}")]
    IncompatiblePreparedFilter { mismatch: PreparedFilterMismatch },
    #[error("Input/output length ({actual}) should be a multiple of ({partition_len})")]
    InvalidInputOutputLen { actual: usize, partition_len: usize },
    #[error("The owls are not what they seem")]
    InternalProcessingError(#[from] realfft::FftError),
    #[error("Filter delay ({delay_samples} samples) exceeds the max delay ({max_samples} samples)")]
    DelayExceedsCapacity {
        delay_samples: usize,
        max_samples: usize,
    },
}

/// A rejected update, retaining ownership so the caller can reclaim it safely.
///
/// The message already includes the reason, so
/// [`source`](std::error::Error::source) returns `None` to avoid repeating it
/// in error reports.
#[derive(Debug, thiserror::Error)]
#[error("Filter update rejected: {reason}")]
pub struct FilterUpdateError {
    /// Validation or connection failure that rejected the update.
    pub reason: Error,
    /// Rejected owner, retained for off-thread reclamation or retry.
    pub filter: PreparedFilter,
}

/// Envelope shape used when crossfading between two filters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FadeCurve {
    /// Linear constant-amplitude fade.
    Linear,
    /// Cosine-square constant-amplitude fade. This is the default.
    #[default]
    CosineSquared,
}

impl FadeCurve {
    fn gains(self, position: usize, len: usize) -> (f32, f32) {
        if len <= 1 || position >= len {
            return (0.0, 1.0);
        }

        let t = position as f32 / (len - 1) as f32;

        let fade_in = match self {
            Self::Linear => t,
            Self::CosineSquared => (std::f32::consts::FRAC_PI_2 * t).sin().powi(2),
        };

        (1.0 - fade_in, fade_in)
    }
}

/// How a renderer replaces an installed filter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FilterTransition {
    /// Replace immediately, without allocating crossfade-only state.
    #[default]
    Immediate,
    /// Crossfade rendered outputs over exactly `duration_samples` samples.
    ///
    /// Target delay lines are primed before the envelope begins. An ongoing
    /// transition finishes before the latest queued update starts, at the next
    /// partition boundary. The first filter always installs immediately.
    Crossfade {
        /// Nonzero envelope length in output samples, excluding warmup.
        duration_samples: usize,
        /// Gain envelope applied to the old and new outputs.
        curve: FadeCurve,
    },
}

impl FilterTransition {
    /// Crossfade over `duration_samples` output samples with the default
    /// [`FadeCurve`].
    ///
    /// Construct [`Crossfade`](Self::Crossfade) directly to choose the curve.
    pub const fn crossfade(duration_samples: usize) -> Self {
        Self::Crossfade {
            duration_samples,
            curve: FadeCurve::CosineSquared,
        }
    }
}

/// Whether to apply the per-ear delays carried by each filter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DelayMode {
    /// Ignore filter delays. This preserves the default renderer behavior.
    #[default]
    Disabled,
    /// Preallocate both delay lines for up to `max_samples` samples each.
    ///
    /// Capacity is independent of FIR length. Updates exceeding it are
    /// rejected before changing renderer state; processing never resizes it.
    FromFilter {
        /// Inclusive delay limit per ear, in samples; zero permits only zero delay.
        max_samples: usize,
    },
}

/// Per-source behavior, independent of the shared FFT plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RendererOptions {
    /// Replacement policy for filters after the first installation.
    transition: FilterTransition,
    /// Whether filter delays are applied, and their preallocated capacity.
    delays: DelayMode,
}

impl RendererOptions {
    /// Reject zero-length crossfades and delay lines too large to allocate.
    fn validate(self) -> Result<(), Error> {
        if let FilterTransition::Crossfade {
            duration_samples: 0,
            ..
        } = self.transition
        {
            return Err(Error::InvalidCrossfadeDuration);
        }

        // The circular delay buffer needs one slot beyond the maximum delay.
        if let DelayMode::FromFilter { max_samples } = self.delays
            && max_samples >= isize::MAX as usize / size_of::<f32>()
        {
            return Err(Error::DelayCapacityTooLarge { max_samples });
        }

        Ok(())
    }

    /// Check that a prepared filter matches the plan and fits the delay lines.
    fn validate_filter(self, plan: &RendererPlan, filter: &PreparedFilter) -> Result<(), Error> {
        plan.validate_filter(filter)?;
        self.validate_delays(filter.delay_samples())
    }

    /// Check per-ear delays, in whole samples, against the delay capacity.
    fn validate_delays(self, delays: [usize; 2]) -> Result<(), Error> {
        let DelayMode::FromFilter { max_samples } = self.delays else {
            return Ok(());
        };

        for delay_samples in delays {
            if delay_samples > max_samples {
                return Err(Error::DelayExceedsCapacity {
                    delay_samples,
                    max_samples,
                });
            }
        }

        Ok(())
    }

    /// Per-ear delay limit in samples, or `None` when filter delays are ignored.
    fn max_delay_samples(self) -> Option<usize> {
        match self.delays {
            DelayMode::Disabled => None,
            DelayMode::FromFilter { max_samples } => Some(max_samples),
        }
    }
}

/// A single source's convolution history and mutable rendering state.
///
/// Create one with [`Renderer::new`], or use [`Renderer::builder`] to enable
/// crossfades or filter delays. To update filters from another thread while
/// an audio callback renders, convert it with
/// [`into_realtime`](Self::into_realtime).
///
/// Construction and cloning allocate. Processing never allocates or frees
/// memory.
#[derive(Clone)]
pub struct Renderer {
    /// Mutable convolution, delay, and transition state for this source.
    engine: Engine,
    /// Reusable stereo partition before copying or adding it to caller output.
    output: StereoBlock,
}

impl Renderer {
    /// Create a renderer that replaces filters immediately and ignores filter
    /// delays.
    ///
    /// Use [`Renderer::builder`] to change these policies.
    pub fn new(plan: &RendererPlan) -> Self {
        Self::with_options(plan.clone(), RendererOptions::default())
    }

    /// Configure crossfades or filter delays for a renderer using `plan`.
    pub fn builder(plan: &RendererPlan) -> RendererBuilder {
        RendererBuilder::new(plan)
    }

    /// Allocate per-source buffers for already validated policies.
    fn with_options(plan: RendererPlan, options: RendererOptions) -> Self {
        let output = stereo_block(plan.partition_len());

        Self {
            engine: Engine::new(plan, options),
            output,
        }
    }

    /// Shared configuration and FFT plans.
    pub fn plan(&self) -> &RendererPlan {
        &self.engine.plan
    }

    /// This renderer's transition and delay policy.
    fn options(&self) -> RendererOptions {
        self.engine.options
    }

    /// Prepare a time-domain filter and install it.
    ///
    /// The filter is validated first; on error the renderer is unchanged.
    ///
    /// To update filters from another thread, use
    /// [`into_realtime`](Self::into_realtime).
    pub fn set_filter(&mut self, filter: &Filter) -> Result<(), Error> {
        self.engine.set_filter(filter)
    }

    /// Transfer a prepared filter into this renderer without allocating.
    ///
    /// Returns a superseded queued filter or a previously retired filter.
    /// During a crossfade the newest update replaces the queued update, not
    /// the current target. Errors retain the rejected filter.
    ///
    /// Custom real-time handoffs must reclaim the returned value (including
    /// errors) off-thread and drain [`take_retired_filter`](Self::take_retired_filter)
    /// after processing. [`RealtimeRenderer`] handles this automatically.
    #[must_use = "returned filter storage must be reclaimed off the audio thread"]
    pub fn set_prepared_filter(
        &mut self,
        filter: PreparedFilter,
    ) -> Result<Option<PreparedFilter>, FilterUpdateError> {
        if let Err(reason) = self.options().validate_filter(self.plan(), &filter) {
            return Err(FilterUpdateError { reason, filter });
        }

        Ok(self.engine.install(filter))
    }

    /// Move one filter retired by completed crossfades to the caller.
    ///
    /// At most two filters await reclamation. This never allocates or destroys
    /// filter storage. Destroy returned values on a non-real-time thread.
    /// Renderers updated only with [`set_filter`](Self::set_filter) need not
    /// call this, because `set_filter` reuses retired filters.
    #[must_use = "retired filter storage must be reclaimed off the audio thread"]
    pub fn take_retired_filter(&mut self) -> Option<PreparedFilter> {
        self.engine.take_retired()
    }

    /// Split this renderer into a [`FilterPublisher`] for a worker thread and
    /// a [`RealtimeRenderer`] for the audio callback.
    ///
    /// The renderer keeps its policies, installed filters, and processing
    /// history. Conversion allocates the update queues and the publisher's
    /// preparation buffers, so call it outside the audio callback.
    ///
    /// ```
    /// use sofar::filter::Filter;
    /// use sofar::render::{FilterTransition, Renderer, RendererPlan};
    ///
    /// let plan = RendererPlan::builder(128).with_partition_len(64).build()?;
    ///
    /// let (mut publisher, mut renderer) = Renderer::builder(&plan)
    ///     .with_filter_transition(FilterTransition::crossfade(32))
    ///     .with_max_delay(0.005)
    ///     .build()?
    ///     .into_realtime();
    ///
    /// // Worker thread: prepare and publish filters, e.g. from `Sofar::filter`.
    /// publisher.publish_filter(&Filter::new(128))?;
    ///
    /// // Audio callback: adopts the latest filter and returns displaced ones.
    /// renderer.process_block(&[0.0; 64], &mut [0.0; 64], &mut [0.0; 64])?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_realtime(self) -> (FilterPublisher, RealtimeRenderer) {
        RealtimeRenderer::new(self)
    }

    /// Render mono input into stereo output, replacing existing output samples.
    ///
    /// The input length must be a multiple of the plan's partition length.
    ///
    /// # Panics
    ///
    /// Panics if either output length differs from the input length.
    pub fn process_block(
        &mut self,
        input: &[f32],
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        self.process(input, left, right, OutputMode::Overwrite, |_| {})
    }

    /// Render mono input and add it to an existing stereo output bus.
    ///
    /// Length requirements and panics are the same as [`Self::process_block`].
    pub fn process_block_add(
        &mut self,
        input: &[f32],
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        self.process(input, left, right, OutputMode::Add, |_| {})
    }

    /// Clear input history and delay lines, retaining installed/queued filters.
    ///
    /// An ongoing transition restarts, including delay warmup. This does not
    /// allocate or destroy filter storage.
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Render whole partitions of `input` into `left` and `right`.
    ///
    /// 1. Check that both outputs match the input length and that the input
    ///    holds a whole number of partitions.
    /// 2. Before each partition, let `before_partition` apply pending updates
    ///    at the partition boundary.
    /// 3. Convolve the partition into the reusable stereo block, then copy or
    ///    add it to the caller's output according to `mode`.
    fn process(
        &mut self,
        input: &[f32],
        left: &mut [f32],
        right: &mut [f32],
        mode: OutputMode,
        mut before_partition: impl FnMut(&mut Engine),
    ) -> Result<(), Error> {
        assert_eq!(left.len(), input.len());
        assert_eq!(right.len(), input.len());

        let partition_len = self.plan().partition_len();

        if !input.len().is_multiple_of(partition_len) {
            return Err(Error::InvalidInputOutputLen {
                actual: input.len(),
                partition_len,
            });
        }

        // Apply updates at partition boundaries, reusing one stereo scratch block.
        for ((input, left), right) in input
            .chunks_exact(partition_len)
            .zip(left.chunks_exact_mut(partition_len))
            .zip(right.chunks_exact_mut(partition_len))
        {
            before_partition(&mut self.engine);
            self.engine.process_partition(input, &mut self.output)?;

            mode.write(left, &self.output[0]);
            mode.write(right, &self.output[1]);
        }

        Ok(())
    }
}

impl fmt::Debug for Renderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let options = self.options();

        f.debug_struct("Renderer")
            .field("plan", self.plan())
            .field("transition", &options.transition)
            .field("max_delay_samples", &options.max_delay_samples())
            .field("filter", &self.engine.filter())
            .finish_non_exhaustive()
    }
}

/// Whether a rendered partition replaces or accumulates into output samples.
#[derive(Clone, Copy)]
enum OutputMode {
    Overwrite,
    Add,
}

impl OutputMode {
    fn write(self, output: &mut [f32], rendered: &[f32]) {
        match self {
            Self::Overwrite => output.copy_from_slice(rendered),
            Self::Add => {
                for (output, sample) in output.iter_mut().zip(rendered) {
                    *output += sample;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
