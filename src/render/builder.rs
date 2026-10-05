use super::filter::delay_capacity;
use super::{DelayMode, Error, FilterTransition, Renderer, RendererOptions, RendererPlan};

/// Per-source renderer configuration backed by an existing immutable FFT plan.
///
/// Start with [`Renderer::builder`]. By default, filters are replaced
/// immediately and filter delays are ignored, as with [`Renderer::new`].
///
/// ```
/// use sofar::render::{FilterTransition, Renderer, RendererPlan};
///
/// let plan = RendererPlan::builder(128).with_partition_len(64).build()?;
///
/// let renderer = Renderer::builder(&plan)
///     .with_filter_transition(FilterTransition::crossfade(32))
///     .with_max_delay(0.005)
///     .build()?;
/// # Ok::<(), sofar::render::Error>(())
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct RendererBuilder {
    /// Shared validated plan, cloned from the builder's input.
    plan: RendererPlan,
    /// Replacement policy for filters after the first installation.
    transition: FilterTransition,
    /// Per-ear filter delay limit in seconds, or `None` to ignore filter delays.
    max_delay: Option<f32>,
}

impl RendererBuilder {
    /// Retain the shared plan and start with default per-source policies.
    pub(super) fn new(plan: &RendererPlan) -> Self {
        Self {
            plan: plan.clone(),
            transition: FilterTransition::default(),
            max_delay: None,
        }
    }

    /// Set filter-replacement behavior. Defaults to
    /// [`FilterTransition::Immediate`].
    pub fn with_filter_transition(mut self, transition: FilterTransition) -> Self {
        self.transition = transition;

        self
    }

    /// Apply the per-ear delays carried by each filter, up to `seconds`.
    ///
    /// Without this, filter delays are ignored. Both delay lines are allocated
    /// once, for `seconds` rounded up to whole samples at the plan's sample
    /// rate. Filters with longer delays are rejected with
    /// [`Error::DelayExceedsCapacity`] and leave the renderer unchanged.
    ///
    /// [`Sofar::max_delay`](crate::reader::Sofar::max_delay) returns a limit
    /// that accepts every filter from a SOFA file.
    pub fn with_max_delay(mut self, seconds: f32) -> Self {
        self.max_delay = Some(seconds);

        self
    }

    /// Validate the policies and allocate the renderer.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidCrossfadeDuration`] if a crossfade lasts zero samples.
    /// - [`Error::InvalidDelay`] if the maximum delay is negative, not finite,
    ///   or too large to represent in samples.
    /// - [`Error::DelayCapacityTooLarge`] if the delay lines would exceed the
    ///   largest supported allocation.
    pub fn build(self) -> Result<Renderer, Error> {
        let delays = match self.max_delay {
            None => DelayMode::Disabled,
            Some(seconds) => DelayMode::FromFilter {
                max_samples: delay_capacity(seconds, self.plan.sample_rate())?,
            },
        };

        let options = RendererOptions {
            transition: self.transition,
            delays,
        };

        options.validate()?;

        Ok(Renderer::with_options(self.plan, options))
    }
}
