use super::*;
use crate::filter::Filter;
use crate::render::{FadeCurve, FilterTransition, PreparedFilterMismatch};

/// Sample rate at which whole-sample delays are exact in seconds.
const SAMPLE_RATE: f32 = 1024.0;

/// Setup for one-tap real-time tests with four-sample partitions.
///
/// Only construction is shared: tests retain explicit ownership of both endpoints
/// and control every publication, processing call, and reclamation.
#[derive(Default)]
#[must_use]
struct RealtimeTest {
    /// Replacement policy, defaulting to immediate updates.
    transition: FilterTransition,
    /// Per-ear delay limit in samples; `None` ignores filter delays.
    max_delay: Option<usize>,
    /// Test-only return queue override; `None` uses `into_realtime`.
    return_capacity: Option<usize>,
}

impl RealtimeTest {
    /// Enable a linear crossfade with the supplied duration in samples.
    fn crossfade(mut self, duration_samples: usize) -> Self {
        self.transition = FilterTransition::Crossfade {
            duration_samples,
            curve: FadeCurve::Linear,
        };

        self
    }

    /// Enable filter delays with an inclusive per-ear sample limit.
    fn max_delay(mut self, max_samples: usize) -> Self {
        self.max_delay = Some(max_samples);

        self
    }

    /// Limit the return queue to exercise backpressure with fewer updates.
    fn return_capacity(mut self, capacity: usize) -> Self {
        self.return_capacity = Some(capacity);

        self
    }

    /// Build independently owned endpoints without preparing or retaining filters.
    fn build(self) -> (RendererPlan, FilterPublisher, RealtimeRenderer) {
        let plan = plan();
        let mut builder = Renderer::builder(&plan).with_filter_transition(self.transition);

        if let Some(max_samples) = self.max_delay {
            builder = builder.with_max_delay(max_samples as f32 / SAMPLE_RATE);
        }

        let renderer = builder.build().unwrap();

        let (publisher, renderer) = match self.return_capacity {
            Some(capacity) => RealtimeRenderer::with_return_capacity(renderer, capacity),
            None => renderer.into_realtime(),
        };

        (plan, publisher, renderer)
    }
}

/// Build the shared layout used by real-time update scenarios.
fn plan() -> RendererPlan {
    RendererPlan::builder(1)
        .with_sample_rate(SAMPLE_RATE)
        .with_partition_len(4)
        .build()
        .unwrap()
}

/// Build a one-tap stereo filter with the given gain and no delay.
fn gain_filter(gain: f32) -> Filter {
    let mut filter = Filter::new(1);
    filter.left[0] = gain;
    filter.right[0] = gain;

    filter
}

/// Prepare a stereo gain filter without retaining an extra owner in the fixture.
fn prepare(plan: &RendererPlan, gain: f32) -> PreparedFilter {
    plan.filter_transform().prepare(&gain_filter(gain)).unwrap()
}

/// Process one partition of constant input without worker-side reclamation.
fn render(renderer: &mut RealtimeRenderer) -> [f32; 4] {
    let mut left = [0.0; 4];

    renderer
        .process_block(&[1.0; 4], &mut left, &mut [0.0; 4])
        .unwrap();

    left
}

#[test]
fn into_realtime_preserves_configuration_and_installed_filters() {
    let plan = plan();
    let mut renderer = Renderer::builder(&plan)
        .with_filter_transition(FilterTransition::Crossfade {
            duration_samples: 8,
            curve: FadeCurve::Linear,
        })
        .with_max_delay(8.0 / SAMPLE_RATE)
        .build()
        .unwrap();
    let options = renderer.options();

    // Leave an active, a fading, and a queued filter, plus a spare.
    for gain in [1.0, 0.5, 0.25, 0.125] {
        renderer.set_filter(&gain_filter(gain)).unwrap();
    }

    assert_eq!(renderer.engine.owned_filters(), 4);

    let (mut publisher, mut renderer) = renderer.into_realtime();

    assert_eq!(publisher.options, options);
    assert_eq!(renderer.renderer.options(), options);
    assert!(Arc::ptr_eq(&plan.0, &publisher.transform.plan.0));
    assert!(Arc::ptr_eq(&plan.0, &renderer.plan().0));

    // The spare is released during conversion instead of on the audio thread.
    assert_eq!(renderer.renderer.engine.owned_filters(), 3);

    // Both eight-sample fades finish within four partitions.
    for _ in 0..4 {
        render(&mut renderer);
    }

    assert_eq!(render(&mut renderer), [0.125; 4]);

    publisher.reclaim();

    assert_eq!(renderer.renderer.engine.owned_filters(), 1);
}

#[test]
fn publish_filter_prepares_validates_and_applies_updates() {
    let (_, mut publisher, mut renderer) = RealtimeTest::default().max_delay(8).build();

    publisher.publish_filter(&gain_filter(0.5)).unwrap();

    assert_eq!(render(&mut renderer), [0.5; 4]);

    let mut delayed = gain_filter(1.0);
    delayed.ldelay = 9.0 / SAMPLE_RATE;

    assert!(matches!(
        publisher.publish_filter(&delayed),
        Err(Error::DelayExceedsCapacity {
            delay_samples: 9,
            max_samples: 8
        })
    ));
    assert!(matches!(
        publisher.publish_filter(&Filter::new(2)),
        Err(Error::InvalidFilterLength {
            actual: 2,
            expected: 1
        })
    ));

    delayed.ldelay = 8.0 / SAMPLE_RATE;
    publisher.publish_filter(&delayed).unwrap();

    // The accepted delayed filter replaces the active one immediately.
    assert_eq!(render(&mut renderer), [0.0; 4]);
}

#[test]
fn publisher_reports_disconnection_before_preparing() {
    let (_, mut publisher, renderer) = RealtimeTest::default().build();

    assert!(publisher.is_connected());

    drop(renderer);

    assert!(!publisher.is_connected());

    // The invalid length would fail preparation, but disconnection is checked first.
    assert!(matches!(
        publisher.publish_filter(&Filter::new(2)),
        Err(Error::RendererDisconnected)
    ));
}

#[test]
fn only_latest_pending_filter_is_applied_and_replaced_owner_is_reclaimed() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default().build();

    let first = prepare(&plan, 1.0);
    let retired = Arc::downgrade(&first.0);

    publisher.publish(first).unwrap();
    publisher.publish(prepare(&plan, 0.5)).unwrap();

    assert!(retired.upgrade().is_none());
    assert_eq!(render(&mut renderer), [0.5; 4]);
    assert_eq!(render(&mut renderer), [0.5; 4]);
    assert!(publisher.returned.is_empty());
}

#[test]
fn full_return_queue_retains_latest_pending_update_and_keeps_rendering() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default().return_capacity(1).build();

    publisher.publish(prepare(&plan, 1.0)).unwrap();
    assert_eq!(render(&mut renderer), [1.0; 4]);

    publisher.publish(prepare(&plan, 0.5)).unwrap();
    assert_eq!(render(&mut renderer), [0.5; 4]);
    assert!(publisher.returned.is_full());

    // Keep the return queue full, as if the worker published before the
    // callback returned the retired filter.
    drop(publisher.mailbox.replace(Some(prepare(&plan, 0.25))));
    assert_eq!(render(&mut renderer), [0.5; 4]);
    assert!(publisher.returned.is_full());

    drop(publisher.mailbox.replace(Some(prepare(&plan, 0.125))));
    publisher.reclaim();

    assert_eq!(render(&mut renderer), [0.125; 4]);
}

#[test]
fn crossfades_return_retired_filters_even_without_a_new_update() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default().crossfade(8).build();

    let first = prepare(&plan, 1.0);
    let first_owner = Arc::downgrade(&first.0);

    publisher.publish(first).unwrap();
    render(&mut renderer);

    publisher.publish(prepare(&plan, 0.5)).unwrap();
    render(&mut renderer);

    publisher.publish(prepare(&plan, 0.25)).unwrap();
    render(&mut renderer);

    assert_eq!(publisher.returned.occupied_len(), 1);
    assert_eq!(first_owner.strong_count(), 1);

    publisher.reclaim();
    assert!(first_owner.upgrade().is_none());

    // The queued fade must retire its old owner without another publication.
    render(&mut renderer);
    render(&mut renderer);

    assert_eq!(publisher.returned.occupied_len(), 1);
}

#[test]
fn queued_crossfade_can_finish_while_the_return_queue_is_full() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default()
        .crossfade(8)
        .return_capacity(1)
        .build();

    for gain in [1.0, 0.5, 0.25] {
        publisher.publish(prepare(&plan, gain)).unwrap();
        render(&mut renderer);
    }

    assert!(publisher.returned.is_full());
    render(&mut renderer);
    assert_eq!(render(&mut renderer)[3], 0.25);

    drop(publisher.mailbox.replace(Some(prepare(&plan, 0.125))));
    assert_eq!(render(&mut renderer), [0.25; 4]);

    publisher.reclaim();
    assert_eq!(render(&mut renderer), [0.25; 4]);
    assert!(publisher.returned.is_full());

    publisher.reclaim();
    render(&mut renderer);
    assert_eq!(render(&mut renderer)[3], 0.125);
}

#[test]
fn rejected_updates_leave_the_valid_pending_owner_untouched() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default().max_delay(8).build();

    publisher.publish(prepare(&plan, 0.5)).unwrap();

    let incompatible_plan = RendererPlan::builder(1)
        .with_partition_len(8)
        .build()
        .unwrap();
    let error = publisher
        .publish(prepare(&incompatible_plan, 1.0))
        .unwrap_err();

    assert!(matches!(
        error.reason,
        Error::IncompatiblePreparedFilter {
            mismatch: PreparedFilterMismatch::PartitionLength {
                filter: 8,
                renderer: 4
            }
        }
    ));

    let mut delayed = Filter::new(1);
    delayed.ldelay = 9.0 / plan.sample_rate();

    let error = publisher
        .publish(plan.filter_transform().prepare(&delayed).unwrap())
        .unwrap_err();

    assert!(matches!(error.reason, Error::DelayExceedsCapacity { .. }));
    assert_eq!(render(&mut renderer), [0.5; 4]);
}

#[test]
fn invalid_audio_block_does_not_consume_an_update() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default().build();

    publisher.publish(prepare(&plan, 0.5)).unwrap();

    assert!(
        renderer
            .process_block(&[1.0; 3], &mut [0.0; 3], &mut [0.0; 3])
            .is_err()
    );

    assert_eq!(render(&mut renderer), [0.5; 4]);
}

#[test]
fn either_shutdown_order_releases_all_filter_storage() {
    for publisher_first in [false, true] {
        let (plan, mut publisher, mut renderer) = RealtimeTest::default().build();

        let first = prepare(&plan, 1.0);
        let second = prepare(&plan, 0.5);
        let owners = [Arc::downgrade(&first.0), Arc::downgrade(&second.0)];

        publisher.publish(first).unwrap();
        render(&mut renderer);
        publisher.publish(second).unwrap();

        if publisher_first {
            drop(publisher);
            render(&mut renderer);
            drop(renderer);
        } else {
            drop(renderer);

            let error = publisher.publish(prepare(&plan, 0.25)).unwrap_err();

            assert!(matches!(error.reason, Error::RendererDisconnected));
            drop(publisher);
        }

        assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    }
}

#[test]
fn concurrent_publication_transfers_each_owner_exactly_once() {
    let (plan, mut publisher, mut renderer) = RealtimeTest::default().build();

    let filters: Vec<_> = (0..500).map(|i| prepare(&plan, i as f32)).collect();
    let owners: Vec<_> = filters
        .iter()
        .map(|filter| Arc::downgrade(&filter.0))
        .collect();

    let worker = std::thread::spawn(move || {
        for filter in filters {
            publisher.publish(filter).unwrap();
            std::thread::yield_now();
        }

        publisher
    });

    while !worker.is_finished() {
        render(&mut renderer);
        std::thread::yield_now();
    }

    let mut publisher = worker.join().unwrap();

    // Reclamation makes room to adopt the final pending update.
    publisher.reclaim();
    assert_eq!(render(&mut renderer), [499.0; 4]);

    publisher.reclaim();
    drop(renderer);
    drop(publisher);

    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
}
