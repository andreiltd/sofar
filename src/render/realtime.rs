use std::fmt;
use std::ptr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicPtr, Ordering},
};

use ringbuf::{
    HeapCons, HeapProd, HeapRb,
    traits::{Consumer, Observer, Producer, Split},
};

use crate::filter::Filter;

use super::convolution::Engine;
use super::filter::PreparedFilterData;
use super::{
    Error, FilterTransform, FilterUpdateError, OutputMode, PreparedFilter, Renderer,
    RendererOptions, RendererPlan,
};

/// Filters the renderer can return before the publisher must reclaim them.
const RETURN_CAPACITY: usize = 8;

/// Worker-side endpoint that updates one [`RealtimeRenderer`].
///
/// Create both endpoints with [`Renderer::into_realtime`]. Publishing hands a
/// filter to the renderer, which adopts it at its next partition boundary.
/// Filters the renderer displaces come back to this endpoint, which frees them
/// when publishing or reclaiming, so the audio thread never frees memory.
///
/// Updates are latest-wins: publishing replaces an update the renderer has not
/// adopted yet. An ongoing crossfade finishes before the latest update starts.
///
/// Use this endpoint outside the audio callback. Dropping it is safe at any
/// time; the renderer keeps rendering its current filters and frees returned
/// filters when it is dropped.
pub struct FilterPublisher {
    /// FFT workspace for the bound renderer's plan.
    transform: FilterTransform,
    /// Bound renderer's policies, including its delay-capacity limit.
    options: RendererOptions,
    /// Latest-update slot; replaced owners are reclaimed by the publisher.
    mailbox: Arc<Mailbox>,
    /// Retired-owner queue consumer, drained off the audio thread.
    returned: HeapCons<PreparedFilter>,
}

impl FilterPublisher {
    /// Prepare a time-domain filter and publish it to the renderer.
    ///
    /// This allocates the prepared filter and frees filters the renderer has
    /// returned. Nothing is published on error.
    ///
    /// # Errors
    ///
    /// - [`Error::RendererDisconnected`] if the renderer has been dropped. The
    ///   filter is not prepared.
    /// - The errors of [`FilterTransform::prepare`] for invalid filters.
    /// - [`Error::DelayExceedsCapacity`] if a delay exceeds the renderer's
    ///   maximum delay.
    pub fn publish_filter(&mut self, filter: &Filter) -> Result<(), Error> {
        self.reclaim();

        // Skip preparation when no renderer can adopt the filter.
        if !self.is_connected() {
            return Err(Error::RendererDisconnected);
        }

        let filter = self.transform.prepare(filter)?;

        self.publish(filter).map_err(|error| error.reason)
    }

    /// Validate and publish a filter that is already prepared, for example
    /// one cached or prepared by a separate [`FilterTransform`].
    ///
    /// This also reclaims returned filters. Invalid updates retain ownership
    /// in the error and leave the existing pending update unchanged.
    pub fn publish(&mut self, filter: PreparedFilter) -> Result<(), FilterUpdateError> {
        self.reclaim();

        if !self.is_connected() {
            return Err(FilterUpdateError {
                reason: Error::RendererDisconnected,
                filter,
            });
        }

        if let Err(reason) = self.options.validate_filter(&self.transform.plan, &filter) {
            return Err(FilterUpdateError { reason, filter });
        }

        // Replacing the slot reclaims an unconsumed update on this worker thread.
        drop(self.mailbox.replace(Some(filter)));

        Ok(())
    }

    /// Free filters returned by the renderer on this non-real-time thread.
    ///
    /// Publishing also reclaims. Call this periodically when no further
    /// filters are being published.
    pub fn reclaim(&mut self) {
        while self.returned.try_pop().is_some() {}
    }

    /// Whether the bound [`RealtimeRenderer`] still exists.
    ///
    /// Once this returns `false`, publishing fails with
    /// [`Error::RendererDisconnected`].
    pub fn is_connected(&self) -> bool {
        self.mailbox.connected.load(Ordering::Acquire)
    }
}

impl fmt::Debug for FilterPublisher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterPublisher")
            .field("plan", &self.transform.plan)
            .field("connected", &self.is_connected())
            .finish_non_exhaustive()
    }
}

/// Renderer that adopts filters from its bound [`FilterPublisher`].
///
/// Create both endpoints with [`Renderer::into_realtime`]. Processing adopts
/// the latest published filter at partition boundaries and sends displaced
/// filters back to the publisher. It never allocates or frees memory,
/// including its first call on a fresh thread. If returned filters have not
/// been reclaimed yet, adoption is deferred without interrupting rendering.
///
/// This type intentionally does not expose mutable access to the underlying
/// renderer. Drop it outside the audio callback, after stopping audio processing.
pub struct RealtimeRenderer {
    /// Per-source DSP state with preallocated processing buffers.
    renderer: Renderer,
    /// Bound mailbox consumer and retirement-queue producer.
    updates: RtUpdates,
}

impl RealtimeRenderer {
    /// Bind `renderer` to a new publisher.
    pub(super) fn new(renderer: Renderer) -> (FilterPublisher, Self) {
        Self::with_return_capacity(renderer, RETURN_CAPACITY)
    }

    /// Bind `renderer` to a new publisher that can hold `return_capacity`
    /// returned filters.
    ///
    /// 1. Drop spare and retired filters now, rather than leaving them for
    ///    the audio thread to return.
    /// 2. Allocate the shared mailbox and the return queue.
    /// 3. Give the publisher its own FFT workspace and the renderer's policies.
    fn with_return_capacity(
        mut renderer: Renderer,
        return_capacity: usize,
    ) -> (FilterPublisher, Self) {
        assert!(return_capacity > 0);

        renderer.engine.release_unused();

        let mailbox = Arc::new(Mailbox {
            pending: AtomicPtr::new(ptr::null_mut()),
            connected: AtomicBool::new(true),
        });
        let (returned, returns) = HeapRb::new(return_capacity).split();

        let publisher = FilterPublisher {
            transform: renderer.plan().filter_transform(),
            options: renderer.options(),
            mailbox: Arc::clone(&mailbox),
            returned: returns,
        };

        let renderer = Self {
            renderer,
            updates: RtUpdates { mailbox, returned },
        };

        (publisher, renderer)
    }

    /// Shared configuration and FFT plans.
    pub fn plan(&self) -> &RendererPlan {
        self.renderer.plan()
    }

    /// Apply pending updates and render mono input into stereo output.
    ///
    /// Length requirements and panics are the same as [`Renderer::process_block`].
    pub fn process_block(
        &mut self,
        input: &[f32],
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        self.process(input, left, right, OutputMode::Overwrite)
    }

    /// Apply pending updates and accumulate rendered samples into a stereo bus.
    ///
    /// Length requirements and panics are the same as [`Renderer::process_block`].
    pub fn process_block_add(
        &mut self,
        input: &[f32],
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        self.process(input, left, right, OutputMode::Add)
    }

    /// Clear processing history without discarding filters or pending updates.
    ///
    /// An active transition restarts as described by [`Renderer::reset`].
    pub fn reset(&mut self) {
        self.renderer.reset();
    }

    /// Render with updates applied at partition boundaries, then return
    /// filters retired by the final partition.
    fn process(
        &mut self,
        input: &[f32],
        left: &mut [f32],
        right: &mut [f32],
        mode: OutputMode,
    ) -> Result<(), Error> {
        let updates = &mut self.updates;
        let result = self
            .renderer
            .process(input, left, right, mode, |engine| updates.apply(engine));

        // Return final-partition owners while space remains, even without another callback.
        updates.return_retired(&mut self.renderer.engine);

        result
    }
}

impl fmt::Debug for RealtimeRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeRenderer")
            .field("renderer", &self.renderer)
            .finish_non_exhaustive()
    }
}

impl Drop for RealtimeRenderer {
    fn drop(&mut self) {
        self.updates
            .mailbox
            .connected
            .store(false, Ordering::Release);
    }
}

/// Audio-thread endpoint for adopting filters and returning retired owners.
struct RtUpdates {
    /// Latest published owner, taken only when return capacity is available.
    mailbox: Arc<Mailbox>,
    /// Queue producer transferring retired owners back to the worker.
    returned: HeapProd<PreparedFilter>,
}

impl RtUpdates {
    /// Return retired owners before reserving space to adopt a pending filter.
    fn apply(&mut self, engine: &mut Engine) {
        self.return_retired(engine);

        // Leave the pending owner in the mailbox unless a displaced owner can be returned.
        if self.returned.is_full() {
            return;
        }

        let Some(filter) = self.mailbox.replace(None) else {
            return;
        };

        let Some(retired) = engine.install(filter) else {
            return;
        };

        self.returned
            .try_push(retired)
            .expect("retirement capacity was reserved before taking the update");
    }

    fn return_retired(&mut self, engine: &mut Engine) {
        while !self.returned.is_full() {
            let Some(filter) = engine.take_retired() else {
                break;
            };

            self.returned
                .try_push(filter)
                .expect("retirement capacity was checked");
        }
    }
}

/// Single-slot, latest-wins transfer of a prepared filter's owning `Arc`.
///
/// Each non-null pointer owns one strong reference, transferred only by swapping.
/// Readers never dereference a borrowed pointer.
struct Mailbox {
    /// Null when empty; otherwise owns one `Arc<PreparedFilterData>` reference.
    pending: AtomicPtr<PreparedFilterData>,
    /// Cleared when the bound real-time renderer begins dropping.
    connected: AtomicBool,
}

impl Mailbox {
    /// Transfer the incoming owner's strong reference into the slot and take the old one.
    fn replace(&self, filter: Option<PreparedFilter>) -> Option<PreparedFilter> {
        let new = filter.map_or(ptr::null_mut(), |filter| Arc::into_raw(filter.0).cast_mut());
        let old = self.pending.swap(new, Ordering::AcqRel);

        if old.is_null() {
            return None;
        }

        // SAFETY: swap transferred the slot's unique strong reference to
        // this caller. Every stored pointer came from Arc::into_raw above.
        Some(PreparedFilter(unsafe { Arc::from_raw(old) }))
    }
}

impl Drop for Mailbox {
    fn drop(&mut self) {
        drop(self.replace(None));
    }
}

#[cfg(test)]
mod tests;
