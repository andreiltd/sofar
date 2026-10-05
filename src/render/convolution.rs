use std::sync::Arc;

use arrayvec::ArrayVec;
use realfft::num_complex::Complex;

use crate::filter::Filter;

use super::plan::PlanData;
use super::{
    DelayMode, Error, FadeCurve, FilterTransition, PreparedFilter, RendererOptions, RendererPlan,
};

/// Planar stereo samples for one processing partition.
pub(super) type StereoBlock = [Box<[f32]>; 2];

pub(super) fn stereo_block(len: usize) -> StereoBlock {
    std::array::from_fn(|_| vec![0.0; len].into_boxed_slice())
}

/// Per-source convolution state, crossfade scheduling, and retired filter owners.
///
/// The active, target, queued, and retired filters never number more than
/// three. With the spare kept by [`set_filter`](Self::set_filter), an engine
/// holds at most four filters, or two with immediate replacement.
#[derive(Clone)]
pub(super) struct Engine {
    /// Shared FFT plans and validated convolution dimensions.
    pub plan: RendererPlan,
    /// Fixed per-source transition and delay policies.
    pub options: RendererOptions,
    /// Input history and scratch reused by active and target filters.
    convolution: Convolution,
    /// Currently active filter path; silent until a filter is installed.
    active: RenderPath,
    /// Preallocated transition state, present only when crossfading is enabled.
    crossfade: Option<Crossfade>,
    /// Up to two retired filters retained until the caller reclaims them.
    retired: ArrayVec<PreparedFilter, 2>,
    /// Displaced filter kept by `set_filter` as storage for the next update.
    spare: Option<PreparedFilter>,
}

impl Engine {
    pub fn new(plan: RendererPlan, options: RendererOptions) -> Self {
        let crossfade = match options.transition {
            FilterTransition::Immediate => None,
            FilterTransition::Crossfade {
                duration_samples,
                curve,
            } => Some(Crossfade {
                target: RenderPath::new(options.delays),
                output: stereo_block(plan.partition_len()),
                phase: Phase::Steady,
                queued: None,
                duration_samples,
                curve,
            }),
        };

        Self {
            convolution: Convolution::new(&plan.0),
            active: RenderPath::new(options.delays),
            crossfade,
            retired: ArrayVec::new(),
            spare: None,
            plan,
            options,
        }
    }

    /// Currently active filter, or `None` before the first installation.
    pub fn filter(&self) -> Option<&PreparedFilter> {
        self.active.filter.as_ref()
    }

    /// Prepare `filter` in reusable storage and install it.
    ///
    /// 1. Validate both FIR lengths, delays, and the delay capacity before
    ///    changing any state.
    /// 2. Take unshared storage from the spare or retired filters, allocating
    ///    only when none is available.
    /// 3. Transform the filter into that storage using the convolution's FFT
    ///    workspaces, which hold no state between partitions.
    /// 4. Install it, keeping the displaced filter as the next spare.
    pub fn set_filter(&mut self, filter: &Filter) -> Result<(), Error> {
        let delays = self.plan.filter_delays(filter)?;

        self.options.validate_delays(delays)?;

        let mut storage = self
            .take_reusable()
            .unwrap_or_else(|| self.plan.allocate_filter());

        let transformed = self.plan.transform_filter(
            filter,
            delays,
            &mut storage,
            &mut self.convolution.scratch,
            &mut self.convolution.rfft_scratch,
        );

        if let Err(error) = transformed {
            self.spare = Some(storage);

            return Err(error);
        }

        // `take_reusable` emptied the spare slot, so no storage is dropped here.
        debug_assert!(self.spare.is_none());
        self.spare = self.install(storage);

        Ok(())
    }

    /// Take the first unshared spare or retired filter.
    ///
    /// Shared filters cannot be overwritten. Dropping this engine's handle to
    /// them leaves their storage to the other owners.
    fn take_reusable(&mut self) -> Option<PreparedFilter> {
        while let Some(mut filter) = self.spare.take().or_else(|| self.retired.pop()) {
            if Arc::get_mut(&mut filter.0).is_some() {
                return Some(filter);
            }
        }

        None
    }

    /// Drop filters kept only for reuse or reclamation.
    ///
    /// Active, target, and queued filters stay installed.
    pub fn release_unused(&mut self) {
        self.spare = None;
        self.retired.clear();
    }

    /// Count filters owned by this engine, including retired and spare ones.
    #[cfg(test)]
    pub fn owned_filters(&self) -> usize {
        let transition = self.crossfade.as_ref().map_or(0, |crossfade| {
            usize::from(crossfade.target.filter.is_some()) + usize::from(crossfade.queued.is_some())
        });

        usize::from(self.active.filter.is_some())
            + transition
            + self.retired.len()
            + usize::from(self.spare.is_some())
    }

    /// Install immediately or queue a fade, returning an owner for reclamation.
    pub fn install(&mut self, filter: PreparedFilter) -> Option<PreparedFilter> {
        if self.active.filter.is_none() || self.crossfade.is_none() {
            self.active.set_delays(filter.delay_samples());

            return self.active.filter.replace(filter);
        }

        let crossfade = self.crossfade.as_mut().expect("crossfade is enabled");

        if crossfade.phase != Phase::Steady {
            // Replacing a queued owner must not also pop a retired owner.
            return crossfade
                .queued
                .replace(filter)
                .or_else(|| self.retired.pop());
        }

        // Only the target and one queued update can finish between installs.
        // Accepting another target releases a retirement slot for its old filter.
        crossfade.start(filter);

        self.retired.pop()
    }

    /// Move one filter retired by a completed crossfade to the caller.
    pub fn take_retired(&mut self) -> Option<PreparedFilter> {
        self.retired.pop()
    }

    /// Render one partition, advance any fade, and retain completed filter owners.
    pub fn process_partition(
        &mut self,
        input: &[f32],
        output: &mut StereoBlock,
    ) -> Result<(), Error> {
        // Advance the input FFT once for both ears and both filter paths.
        self.convolution.prepare_input(&self.plan.0, input)?;

        self.active
            .render(&mut self.convolution, &self.plan.0, output)?;

        let Some(crossfade) = self.crossfade.as_mut() else {
            return Ok(());
        };

        if crossfade.phase == Phase::Steady {
            return Ok(());
        }

        // The target shares input history but fills its own delay lines.
        crossfade
            .target
            .render(&mut self.convolution, &self.plan.0, &mut crossfade.output)?;

        crossfade.mix(output);

        if !crossfade.finished() {
            return Ok(());
        }

        // Promote the target without destroying the old filter on the audio thread.
        std::mem::swap(&mut self.active, &mut crossfade.target);

        self.retired.push(
            crossfade
                .target
                .filter
                .take()
                .expect("completed fade has an old filter"),
        );

        crossfade.phase = Phase::Steady;

        let Some(queued) = crossfade.queued.take() else {
            return Ok(());
        };

        // The queued transition begins rendering at the next partition boundary.
        crossfade.start(queued);

        Ok(())
    }

    pub fn reset(&mut self) {
        self.convolution.reset();
        self.active.reset_delays();

        let Some(crossfade) = self.crossfade.as_mut() else {
            return;
        };

        if crossfade.phase == Phase::Steady {
            return;
        }

        crossfade.prime();
    }
}

/// An active or target stereo filter with independent per-ear delay state.
#[derive(Clone)]
struct RenderPath {
    /// Filter rendered by this path, or `None` for silence.
    filter: Option<PreparedFilter>,
    /// Left/right delay histories; `None` disables filter delays.
    delays: Option<[Delay; 2]>,
}

impl RenderPath {
    fn new(mode: DelayMode) -> Self {
        Self {
            filter: None,
            delays: match mode {
                DelayMode::Disabled => None,
                DelayMode::FromFilter { max_samples } => {
                    Some(std::array::from_fn(|_| Delay::new(max_samples)))
                }
            },
        }
    }

    fn set_delays(&mut self, samples: [usize; 2]) {
        let Some(delays) = &mut self.delays else {
            return;
        };

        for (delay, samples) in delays.iter_mut().zip(samples) {
            delay.set_delay(samples);
        }
    }

    fn reset_delays(&mut self) {
        let Some(delays) = &mut self.delays else {
            return;
        };

        for delay in delays {
            delay.reset();
        }
    }

    fn render(
        &mut self,
        convolution: &mut Convolution,
        plan: &PlanData,
        output: &mut StereoBlock,
    ) -> Result<(), Error> {
        let Some(filter) = &self.filter else {
            for channel in output {
                channel.fill(0.0);
            }

            return Ok(());
        };

        for (spectra, channel) in filter.0.spectra.iter().zip(output.iter_mut()) {
            convolution.apply_filter(plan, spectra, channel)?;
        }

        let Some(delays) = &mut self.delays else {
            return Ok(());
        };

        for (delay, channel) in delays.iter_mut().zip(output) {
            delay.apply(channel);
        }

        Ok(())
    }
}

/// Progress of a filter transition, including target delay-line warmup.
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Steady,
    Warming {
        /// Remaining warmup samples; positive while this phase is active.
        remaining: usize,
    },
    Fading {
        /// Number of mixed fade samples, excluding warmup.
        position: usize,
    },
}

/// Preallocated target path, mix buffers, and the latest queued filter update.
#[derive(Clone)]
struct Crossfade {
    /// Target path, promoted to active when the fade completes.
    target: RenderPath,
    /// Reusable stereo partition rendered through the target path.
    output: StereoBlock,
    /// Current warmup or envelope progress, or idle state.
    phase: Phase,
    /// Latest filter waiting for the current transition to finish.
    queued: Option<PreparedFilter>,
    /// Validated, nonzero envelope length in samples, excluding warmup.
    duration_samples: usize,
    /// Envelope used to blend active and target outputs.
    curve: FadeCurve,
}

impl Crossfade {
    fn start(&mut self, filter: PreparedFilter) {
        assert!(self.target.filter.is_none());

        self.target.filter = Some(filter);
        self.prime();
    }

    /// Restart target delay history before beginning the fade envelope.
    fn prime(&mut self) {
        let delays = self
            .target
            .filter
            .as_ref()
            .expect("transition has a target")
            .delay_samples();

        self.target.reset_delays();
        self.target.set_delays(delays);

        // Both ears must have delayed output ready before the envelope starts.
        let warmup = if self.target.delays.is_some() {
            delays[0].max(delays[1])
        } else {
            0
        };

        self.phase = if warmup == 0 {
            Phase::Fading { position: 0 }
        } else {
            Phase::Warming { remaining: warmup }
        };
    }

    /// Leave warmup samples unchanged, then blend the target sample by sample.
    fn mix(&mut self, output: &mut StereoBlock) {
        for i in 0..output[0].len() {
            match &mut self.phase {
                Phase::Warming { remaining } => {
                    *remaining -= 1;

                    if *remaining == 0 {
                        self.phase = Phase::Fading { position: 0 };
                    }
                }

                Phase::Fading { position } => {
                    if *position >= self.duration_samples {
                        // A fade may end mid-partition; the remaining samples are target-only.
                        for (active, target) in output.iter_mut().zip(&self.output) {
                            active[i..].copy_from_slice(&target[i..]);
                        }

                        return;
                    }

                    let (fade_out, fade_in) = self.curve.gains(*position, self.duration_samples);

                    for (active, target) in output.iter_mut().zip(&self.output) {
                        active[i] = active[i] * fade_out + target[i] * fade_in;
                    }

                    *position = position.saturating_add(1);
                }

                Phase::Steady => unreachable!("only active transitions are mixed"),
            }
        }
    }

    fn finished(&self) -> bool {
        matches!(self.phase, Phase::Fading { position } if position >= self.duration_samples)
    }
}

/// Fixed-capacity per-ear delay line; delay changes never resize its storage.
#[derive(Clone)]
struct Delay {
    /// Circular sample storage; its length is the maximum delay plus one.
    buffer: Box<[f32]>,
    /// Current delay in samples, always smaller than `buffer.len()`.
    delay: usize,
    /// Index of the sample emitted next.
    read: usize,
    /// Index overwritten by the next input sample.
    write: usize,
}

impl Delay {
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples + 1].into_boxed_slice(),
            delay: 0,
            read: 0,
            write: 0,
        }
    }

    fn set_delay(&mut self, delay: usize) {
        assert!(delay < self.buffer.len(), "delay capacity was validated");

        self.delay = delay;
        self.read = (self.write + self.buffer.len() - delay) % self.buffer.len();
    }

    fn apply(&mut self, output: &mut [f32]) {
        for sample in output {
            // Write first so a zero-sample delay passes the current input through.
            self.buffer[self.write] = *sample;
            *sample = self.buffer[self.read];

            self.write = (self.write + 1) % self.buffer.len();
            self.read = (self.read + 1) % self.buffer.len();
        }
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write = 0;

        self.set_delay(self.delay);
    }
}

/// A source's input history and FFT scratch, shared by both ears and fade paths.
#[derive(Clone)]
struct Convolution {
    /// Time-domain workspace for FFT input and inverse-FFT output.
    scratch: Box<[f32]>,
    /// Frequency-bin accumulator for one output channel.
    acc: Box<[Complex<f32>]>,
    /// Forward-FFT scratch reused for input and filter transforms.
    rfft_scratch: Vec<Complex<f32>>,
    /// Inverse-FFT scratch reused across output channels and fade paths.
    ifft_scratch: Vec<Complex<f32>>,
    /// Previous and current input partitions, concatenated for overlap-save.
    x_tdl: Box<[f32]>,
    /// Ring of input spectra, with `spectra_len` contiguous bins per partition.
    x_fdl: Box<[Complex<f32>]>,
    /// Ring slot containing the newest input spectrum.
    fdl_head: usize,
}

impl Convolution {
    fn new(plan: &PlanData) -> Self {
        let zero = Complex::new(0.0, 0.0);

        Self {
            scratch: vec![0.0; plan.fft_len].into_boxed_slice(),
            acc: vec![zero; plan.spectra_len].into_boxed_slice(),
            rfft_scratch: plan.rfft.make_scratch_vec(),
            ifft_scratch: plan.ifft.make_scratch_vec(),
            x_tdl: vec![0.0; plan.fft_len].into_boxed_slice(),
            x_fdl: vec![zero; plan.spectra_data_len].into_boxed_slice(),
            fdl_head: 0,
        }
    }

    /// Assemble an overlap-save window and store its FFT in the input-history ring.
    fn prepare_input(&mut self, plan: &PlanData, input: &[f32]) -> Result<(), Error> {
        let partition_len = plan.layout.partition_len;

        // Retain the preceding partition as overlap for the new input.
        self.x_tdl.copy_within(partition_len.., 0);
        self.x_tdl[partition_len..].copy_from_slice(input);

        // Moving the head backward keeps older spectra at increasing offsets.
        self.fdl_head = if self.fdl_head == 0 {
            plan.partitions - 1
        } else {
            self.fdl_head - 1
        };

        self.scratch.copy_from_slice(&self.x_tdl);
        let offset = self.fdl_head * plan.spectra_len;

        plan.rfft.process_with_scratch(
            &mut self.scratch,
            &mut self.x_fdl[offset..offset + plan.spectra_len],
            &mut self.rfft_scratch,
        )?;

        Ok(())
    }

    /// Sum frequency-domain partitions, invert, and keep the valid output samples.
    fn apply_filter(
        &mut self,
        plan: &PlanData,
        filter: &[Complex<f32>],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.acc.fill(Complex::new(0.0, 0.0));

        // Each FIR partition multiplies the input spectrum from the matching time lag.
        for (p, h) in filter.chunks_exact(plan.spectra_len).enumerate() {
            let offset = (self.fdl_head + p) % plan.partitions * plan.spectra_len;
            let x = &self.x_fdl[offset..offset + plan.spectra_len];

            for (acc, (x, h)) in self.acc.iter_mut().zip(x.iter().zip(h)) {
                *acc += x * h;
            }
        }

        plan.ifft
            .process_with_scratch(&mut self.acc, &mut self.scratch, &mut self.ifft_scratch)?;

        // Overlap-save discards the first half; the inverse FFT also needs normalization.
        for (output, sample) in output
            .iter_mut()
            .zip(&self.scratch[plan.layout.partition_len..])
        {
            *output = sample * plan.inv_scale;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.x_tdl.fill(0.0);
        self.x_fdl.fill(Complex::new(0.0, 0.0));
        self.fdl_head = 0;
    }
}
