use std::sync::Arc;

use assert_approx_eq::assert_approx_eq;

use super::*;

mod cases;

const SAMPLE_RATE: f32 = 1024.0;

/// Build a layout using an exactly representable sample rate for delay tests.
fn plan(filter_len: usize, partition_len: usize) -> RendererPlan {
    RendererPlan::builder(filter_len)
        .with_sample_rate(SAMPLE_RATE)
        .with_partition_len(partition_len)
        .build()
        .unwrap()
}

/// Start a stateful one-tap renderer with four-sample processing partitions.
fn renderer_fixture() -> RendererBuilder {
    Renderer::builder(&plan(1, 4))
}

/// Render a fixed-size buffer without hiding filter updates or owner reclamation.
fn render<const N: usize>(renderer: &mut Renderer, input: [f32; N]) -> [[f32; N]; 2] {
    let [mut left, mut right] = [[0.0; N]; 2];

    renderer
        .process_block(&input, &mut left, &mut right)
        .unwrap();

    [left, right]
}

/// Convert whole samples to seconds; exact because the sample rate is a power of two.
fn seconds(samples: usize) -> f32 {
    samples as f32 / SAMPLE_RATE
}

/// Construct a single nonzero tap with independently specified per-ear delays.
fn tap(filter_len: usize, gain: f32, delays: [usize; 2]) -> Filter {
    let mut filter = Filter::new(filter_len);
    filter.left[0] = gain;
    filter.right[0] = gain;
    filter.ldelay = delays[0] as f32 / SAMPLE_RATE;
    filter.rdelay = delays[1] as f32 / SAMPLE_RATE;

    filter
}

/// Configure a linear envelope with the given duration in samples.
fn crossfade(duration_samples: usize) -> FilterTransition {
    FilterTransition::Crossfade {
        duration_samples,
        curve: FadeCurve::Linear,
    }
}

/// Compare sample buffers with a small magnitude-scaled floating-point tolerance.
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_approx_eq!(actual, expected, 1e-4 * (1.0 + expected.abs()));
    }
}

/// Compute causal time-domain convolution independently of the FFT renderer.
fn convolve(input: &[f32], taps: &[f32], delay: usize) -> Vec<f32> {
    (0..input.len())
        .map(|n| {
            taps.iter()
                .enumerate()
                .filter_map(|(k, tap)| n.checked_sub(k + delay).map(|i| input[i] * tap))
                .sum()
        })
        .collect()
}

#[test]
fn plan_shares_ffts_but_not_history_or_renderer_options() {
    let plan = plan(8, 4);
    let mut first = Renderer::new(&plan);
    let mut second = Renderer::builder(&plan)
        .with_filter_transition(crossfade(8))
        .with_max_delay(seconds(32))
        .build()
        .unwrap();

    assert!(Arc::ptr_eq(&first.plan().0, &second.plan().0));
    assert_ne!(first.options(), second.options());

    let mut filter = Filter::new(8);
    filter.left.fill(1.0);
    filter.right.fill(1.0);

    let prepared = plan.filter_transform().prepare(&filter).unwrap();

    drop(first.set_prepared_filter(prepared.clone()).unwrap());
    drop(second.set_prepared_filter(prepared).unwrap());

    first
        .process_block(&[1.0; 4], &mut [0.0; 4], &mut [0.0; 4])
        .unwrap();

    let mut left = [1.0; 4];

    second
        .process_block(&[0.0; 4], &mut left, &mut [0.0; 4])
        .unwrap();

    assert_close(&left, &[0.0; 4]);
}

#[test]
fn renderers_retain_the_shared_plan_and_default_to_immediate_replacement() {
    let plan = plan(1, 4);
    let filters = [0.5, 0.25].map(|gain| {
        plan.filter_transform()
            .prepare(&tap(1, gain, [8, 9]))
            .unwrap()
    });

    let plan_owner = Arc::downgrade(&plan.0);
    let builder = Renderer::builder(&plan);
    let mut regular = Renderer::new(&plan);
    let (mut publisher, mut realtime) = Renderer::new(&plan).into_realtime();

    // Builders and renderers must retain the plan independently of the caller's handle.
    drop(plan);

    let built = builder.build().unwrap();

    assert!(plan_owner.ptr_eq(&Arc::downgrade(&regular.plan().0)));
    assert!(Arc::ptr_eq(&regular.plan().0, &built.plan().0));
    assert!(Arc::ptr_eq(&regular.plan().0, &realtime.plan().0));
    assert_eq!(regular.options(), RendererOptions::default());
    assert_eq!(built.options(), RendererOptions::default());
    assert_eq!(regular.plan().sample_rate(), SAMPLE_RATE);
    assert_eq!(regular.plan().filter_len(), 1);
    assert_eq!(regular.plan().partition_len(), 4);

    let mut left = [0.0; 4];
    let mut right = [0.0; 4];

    for (gain, filter) in [0.5, 0.25].into_iter().zip(filters) {
        drop(regular.set_prepared_filter(filter.clone()).unwrap());
        publisher.publish(filter).unwrap();

        regular
            .process_block(&[1.0; 4], &mut left, &mut right)
            .unwrap();

        assert_close(&left, &[gain; 4]);
        assert_close(&right, &[gain; 4]);

        realtime
            .process_block(&[1.0; 4], &mut left, &mut right)
            .unwrap();

        assert_close(&left, &[gain; 4]);
        assert_close(&right, &[gain; 4]);
    }
}

#[test]
fn crossfade_shorthand_uses_the_default_curve() {
    assert_eq!(FadeCurve::default(), FadeCurve::CosineSquared);
    assert_eq!(
        FilterTransition::crossfade(32),
        FilterTransition::Crossfade {
            duration_samples: 32,
            curve: FadeCurve::default(),
        }
    );
}

#[test]
fn unconfigured_renderer_outputs_silence_and_keeps_input_history() {
    let plan = plan(8, 4);
    let mut renderer = Renderer::new(&plan);
    let mut left = [9.0; 4];

    renderer
        .process_block(&[1.0; 4], &mut left, &mut [0.0; 4])
        .unwrap();

    assert_close(&left, &[0.0; 4]);

    let mut filter = Filter::new(8);
    filter.left[4] = 1.0;
    renderer.set_filter(&filter).unwrap();

    renderer
        .process_block(&[0.0; 4], &mut left, &mut [0.0; 4])
        .unwrap();

    assert_close(&left, &[1.0; 4]);
}

#[test]
fn oversized_delay_update_retains_ownership_and_preserves_active_filter() {
    let plan = plan(1, 4);
    let mut renderer = Renderer::builder(&plan)
        .with_max_delay(seconds(8))
        .build()
        .unwrap();
    renderer.set_filter(&tap(1, 0.5, [0, 0])).unwrap();

    let filter = plan
        .filter_transform()
        .prepare(&tap(1, 1.0, [9, 0]))
        .unwrap();
    let original = Arc::downgrade(&filter.0);

    let error = renderer.set_prepared_filter(filter).unwrap_err();

    assert!(matches!(
        error.reason,
        Error::DelayExceedsCapacity {
            delay_samples: 9,
            max_samples: 8
        }
    ));
    assert!(original.ptr_eq(&Arc::downgrade(&error.filter.0)));

    let [left, _] = render(&mut renderer, [1.0; 4]);

    assert_close(&left, &[0.5; 4]);
}

#[test]
fn rejected_time_domain_updates_preserve_the_active_filter() {
    let mut renderer = renderer_fixture()
        .with_filter_transition(crossfade(4))
        .with_max_delay(seconds(8))
        .build()
        .unwrap();

    renderer.set_filter(&tap(1, 0.5, [0, 0])).unwrap();

    let mut negative_delay = tap(1, 1.0, [0, 0]);
    negative_delay.rdelay = -1.0;

    for filter in [
        tap(1, 1.0, [9, 0]),
        tap(1, 1.0, [0, 9]),
        tap(2, 1.0, [0, 0]),
        negative_delay,
    ] {
        assert!(renderer.set_filter(&filter).is_err());
    }

    // A rejected update must not start a crossfade or replace the active filter.
    let [left, right] = render(&mut renderer, [1.0; 8]);

    assert_close(&left, &[0.5; 8]);
    assert_close(&right, &[0.5; 8]);
}

#[test]
fn set_filter_never_overwrites_filters_owned_elsewhere() {
    let plan = plan(1, 4);
    let prepared = plan
        .filter_transform()
        .prepare(&tap(1, 0.5, [0, 0]))
        .unwrap();

    let mut renderer = Renderer::new(&plan);
    drop(renderer.set_prepared_filter(prepared.clone()).unwrap());

    // The displaced caller-owned filter becomes the spare but cannot be reused.
    renderer.set_filter(&tap(1, 0.25, [0, 0])).unwrap();
    renderer.set_filter(&tap(1, 0.75, [0, 0])).unwrap();

    // The clone shares the active filter and the spare with the original.
    let mut clone = renderer.clone();

    renderer.set_filter(&tap(1, 1.0, [0, 0])).unwrap();
    renderer.set_filter(&tap(1, 0.125, [0, 0])).unwrap();

    let [left, _] = render(&mut renderer, [1.0; 4]);

    assert_close(&left, &[0.125; 4]);

    let [left, _] = render(&mut clone, [1.0; 4]);

    assert_close(&left, &[0.75; 4]);

    let mut caller = Renderer::new(&plan);
    drop(caller.set_prepared_filter(prepared).unwrap());

    let [left, _] = render(&mut caller, [1.0; 4]);

    assert_close(&left, &[0.5; 4]);
}

#[test]
fn set_filter_keeps_a_bounded_number_of_filters() {
    for (transition, limit) in [(FilterTransition::Immediate, 2), (crossfade(7), 4)] {
        let mut renderer = renderer_fixture()
            .with_filter_transition(transition)
            .build()
            .unwrap();

        for i in 0..300 {
            renderer
                .set_filter(&tap(1, i as f32 / 300.0, [0, 0]))
                .unwrap();

            for _ in 0..i % 5 {
                render(&mut renderer, [1.0; 4]);
            }

            assert!(renderer.engine.owned_filters() <= limit);
        }
    }
}

#[test]
fn filter_update_errors_report_the_reason_once() {
    use std::error::Error as _;

    let plan = plan(1, 4);
    let mut renderer = Renderer::builder(&plan)
        .with_max_delay(seconds(8))
        .build()
        .unwrap();
    let filter = plan
        .filter_transform()
        .prepare(&tap(1, 1.0, [9, 0]))
        .unwrap();

    let error = renderer.set_prepared_filter(filter).unwrap_err();

    assert!(error.source().is_none());
    assert_eq!(
        error.to_string(),
        "Filter update rejected: Filter delay (9 samples) exceeds the max delay (8 samples)"
    );
}

#[test]
fn debug_output_summarizes_layout_and_policies() {
    let plan = plan(1, 4);
    let plan_debug = "RendererPlan { sample_rate: 1024.0, filter_len: 1, partition_len: 4, .. }";

    assert_eq!(format!("{plan:?}"), plan_debug);
    assert_eq!(
        format!("{:?}", RendererPlan::builder(1)),
        "RendererPlanBuilder { sample_rate: 48000.0, filter_len: 1, partition_len: 256 }"
    );
    assert_eq!(
        format!("{:?}", plan.filter_transform()),
        format!("FilterTransform {{ plan: {plan_debug}, .. }}")
    );

    let mut renderer = Renderer::builder(&plan)
        .with_filter_transition(FilterTransition::crossfade(8))
        .with_max_delay(seconds(12))
        .build()
        .unwrap();

    assert_eq!(
        format!("{renderer:?}"),
        format!(
            "Renderer {{ plan: {plan_debug}, \
             transition: Crossfade {{ duration_samples: 8, curve: CosineSquared }}, \
             max_delay_samples: Some(12), filter: None, .. }}"
        )
    );

    renderer.set_filter(&tap(1, 1.0, [5, 9])).unwrap();

    let filter_debug = "PreparedFilter { filter_len: 1, partition_len: 4, \
                        sample_rate: 1024.0, delay_samples: [5, 9], .. }";

    assert!(format!("{renderer:?}").contains(&format!("filter: Some({filter_debug})")));

    let (publisher, realtime) = renderer.into_realtime();

    assert_eq!(
        format!("{publisher:?}"),
        format!("FilterPublisher {{ plan: {plan_debug}, connected: true, .. }}")
    );
    assert!(format!("{realtime:?}").starts_with("RealtimeRenderer { renderer: Renderer { plan: "));

    drop(realtime);

    assert!(format!("{publisher:?}").contains("connected: false"));
}

#[test]
fn invalid_sample_rates_are_rejected() {
    for sample_rate in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        assert!(matches!(
            RendererPlan::builder(1)
                .with_sample_rate(sample_rate)
                .build(),
            Err(Error::InvalidSampleRate { .. })
        ));
    }
}

#[test]
fn invalid_filter_and_partition_lengths_are_rejected() {
    assert!(matches!(
        RendererPlan::builder(0).build(),
        Err(Error::ZeroFilterLength)
    ));

    assert!(matches!(
        RendererPlan::builder(1).with_partition_len(0).build(),
        Err(Error::InvalidPartitionLength { partition_len: 0 })
    ));

    assert!(matches!(
        RendererPlan::builder(1)
            .with_partition_len(usize::MAX)
            .build(),
        Err(Error::LayoutTooLarge)
    ));

    assert!(matches!(
        RendererPlan::builder(usize::MAX).build(),
        Err(Error::LayoutTooLarge)
    ));
}

#[test]
fn zero_crossfade_durations_are_rejected() {
    let result = Renderer::builder(&plan(1, 4))
        .with_filter_transition(crossfade(0))
        .build();

    assert!(matches!(result, Err(Error::InvalidCrossfadeDuration)));
}

#[test]
fn invalid_max_delays_are_rejected() {
    let plan = plan(1, 4);

    for seconds in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(matches!(
            Renderer::builder(&plan).with_max_delay(seconds).build(),
            Err(Error::InvalidDelay { .. })
        ));
    }

    assert!(matches!(
        Renderer::builder(&plan).with_max_delay(1e16).build(),
        Err(Error::DelayCapacityTooLarge { .. })
    ));
}

#[test]
fn max_delay_rounds_up_to_whole_samples() {
    let mut renderer = renderer_fixture()
        .with_max_delay(8.5 / SAMPLE_RATE)
        .build()
        .unwrap();

    assert_eq!(renderer.options().max_delay_samples(), Some(9));

    renderer.set_filter(&tap(1, 1.0, [9, 0])).unwrap();

    assert!(matches!(
        renderer.set_filter(&tap(1, 1.0, [0, 10])),
        Err(Error::DelayExceedsCapacity {
            delay_samples: 10,
            max_samples: 9
        })
    ));

    let zero = renderer_fixture().with_max_delay(0.0).build().unwrap();

    assert_eq!(zero.options().max_delay_samples(), Some(0));
}

#[test]
fn transform_rejects_invalid_delays() {
    let mut transform = plan(1, 4).filter_transform();

    for delay in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut filter = tap(1, 1.0, [0, 0]);
        filter.ldelay = delay;

        assert!(matches!(
            transform.prepare(&filter),
            Err(Error::InvalidDelay { .. })
        ));

        filter.ldelay = 0.0;
        filter.rdelay = delay;

        assert!(matches!(
            transform.prepare(&filter),
            Err(Error::InvalidDelay { .. })
        ));
    }
}

#[test]
fn transform_rejects_either_mismatched_channel() {
    let mut transform = plan(1, 4).filter_transform();

    for (left, right) in [(2, 1), (1, 2)] {
        let filter = Filter {
            left: vec![0.0; left].into_boxed_slice(),
            right: vec![0.0; right].into_boxed_slice(),
            ldelay: 0.0,
            rdelay: 0.0,
        };

        assert!(matches!(
            transform.prepare(&filter),
            Err(Error::InvalidFilterLength {
                actual: 2,
                expected: 1
            })
        ));
    }
}

#[test]
fn prepared_filter_reports_layout_and_structured_mismatches() {
    let source = plan(4, 4);
    let filter = source
        .filter_transform()
        .prepare(&tap(4, 1.0, [8, 9]))
        .unwrap();

    assert_eq!(filter.filter_len(), 4);
    assert_eq!(filter.partition_len(), 4);
    assert_eq!(filter.sample_rate(), SAMPLE_RATE);
    assert_eq!(filter.delay_samples(), [8, 9]);

    for (destination, mismatch) in [
        (
            plan(5, 4),
            PreparedFilterMismatch::FilterLength {
                filter: 4,
                renderer: 5,
            },
        ),
        (
            plan(4, 8),
            PreparedFilterMismatch::PartitionLength {
                filter: 4,
                renderer: 8,
            },
        ),
        (
            RendererPlan::builder(4)
                .with_sample_rate(2.0 * SAMPLE_RATE)
                .with_partition_len(4)
                .build()
                .unwrap(),
            PreparedFilterMismatch::SampleRate {
                filter: SAMPLE_RATE,
                renderer: 2.0 * SAMPLE_RATE,
            },
        ),
    ] {
        let mut renderer = Renderer::new(&destination);
        let error = renderer.set_prepared_filter(filter.clone()).unwrap_err();

        assert!(matches!(
            error.reason,
            Error::IncompatiblePreparedFilter { mismatch: actual } if actual == mismatch
        ));
        assert!(Arc::ptr_eq(&filter.0, &error.filter.0));
    }

    let mut compatible = Renderer::new(&plan(4, 4));

    assert!(compatible.set_prepared_filter(filter).unwrap().is_none());
}

#[test]
fn immediate_replacement_returns_the_complete_stereo_owner() {
    let plan = plan(1, 4);
    let mut transform = plan.filter_transform();
    let first = transform.prepare(&tap(1, 1.0, [8, 9])).unwrap();
    let original = Arc::downgrade(&first.0);
    let second = transform.prepare(&tap(1, 0.5, [10, 11])).unwrap();

    let mut renderer = Renderer::new(&plan);

    assert!(renderer.set_prepared_filter(first).unwrap().is_none());

    let returned = renderer.set_prepared_filter(second).unwrap().unwrap();

    assert!(original.ptr_eq(&Arc::downgrade(&returned.0)));
    assert_eq!(returned.delay_samples(), [8, 9]);

    let returned_again = renderer.set_prepared_filter(returned).unwrap().unwrap();

    assert_eq!(returned_again.delay_samples(), [10, 11]);
}

#[test]
fn fade_curves_have_constant_amplitude_and_exact_endpoints() {
    for curve in [FadeCurve::Linear, FadeCurve::CosineSquared] {
        for len in [1, 2, 32] {
            for position in 0..len + 2 {
                let (out, input) = curve.gains(position, len);
                assert_approx_eq!(out + input, 1.0, 1e-6);
            }

            assert_eq!(curve.gains(len - 1, len), (0.0, 1.0));
        }
    }
}

#[test]
fn fades_match_the_envelope_across_partition_boundaries() {
    for duration_samples in [1, 3, 4, 7, 8, 9] {
        for curve in [FadeCurve::Linear, FadeCurve::CosineSquared] {
            let mut renderer = renderer_fixture()
                .with_filter_transition(FilterTransition::Crossfade {
                    duration_samples,
                    curve,
                })
                .build()
                .unwrap();

            renderer.set_filter(&tap(1, 1.0, [0, 0])).unwrap();
            renderer.set_filter(&tap(1, 0.5, [0, 0])).unwrap();

            let mut left = [0.0; 16];
            let mut right = [0.0; 16];

            for (left, right) in left.chunks_mut(4).zip(right.chunks_mut(4)) {
                renderer.process_block(&[1.0; 4], left, right).unwrap();
            }

            let expected: Vec<_> = (0..16)
                .map(|i| {
                    let (out, input) = curve.gains(i, duration_samples);
                    out + input * 0.5
                })
                .collect();

            assert_close(&left, &expected);
            assert_close(&right, &expected);
        }
    }
}

#[test]
fn mixed_prepared_and_time_domain_updates_keep_only_the_latest_queued_filter() {
    for prepared_first in [false, true] {
        let plan = plan(1, 4);
        let mut transform = plan.filter_transform();
        let mut renderer = Renderer::builder(&plan)
            .with_filter_transition(crossfade(4))
            .build()
            .unwrap();

        // The first update is active, the second is fading, and the last replaces the queue.
        for (i, gain) in [1.0, 0.0, 0.25, 0.5].into_iter().enumerate() {
            let filter = tap(1, gain, [0, 0]);

            if (i % 2 == 0) == prepared_first {
                drop(
                    renderer
                        .set_prepared_filter(transform.prepare(&filter).unwrap())
                        .unwrap(),
                );
            } else {
                renderer.set_filter(&filter).unwrap();
            }
        }

        let [left, _] = render(&mut renderer, [1.0; 12]);

        assert_close(
            &left,
            &[
                1.0,
                2.0 / 3.0,
                1.0 / 3.0,
                0.0,
                0.0,
                1.0 / 6.0,
                1.0 / 3.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        );
    }
}

#[test]
fn crossfade_completion_retains_both_old_filters_without_destruction() {
    let plan = plan(1, 4);
    let mut transform = plan.filter_transform();
    let filters = [1.0, 0.5, 0.25].map(|gain| transform.prepare(&tap(1, gain, [0, 0])).unwrap());
    let old = [Arc::downgrade(&filters[0].0), Arc::downgrade(&filters[1].0)];

    let mut renderer = Renderer::builder(&plan)
        .with_filter_transition(crossfade(4))
        .build()
        .unwrap();

    for filter in filters {
        assert!(renderer.set_prepared_filter(filter).unwrap().is_none());
    }

    render(&mut renderer, [1.0; 8]);

    assert!(old.iter().all(|filter| filter.strong_count() == 1));

    for _ in 0..2 {
        let retired = renderer.take_retired_filter().unwrap();

        assert!(
            old.iter()
                .any(|filter| filter.ptr_eq(&Arc::downgrade(&retired.0)))
        );
    }

    assert!(renderer.take_retired_filter().is_none());
    assert!(old.iter().all(|filter| filter.upgrade().is_none()));
}

#[test]
fn retirement_storage_stays_bounded_when_callers_do_not_drain_between_updates() {
    let plan = plan(1, 4);
    let mut transform = plan.filter_transform();
    let mut renderer = Renderer::builder(&plan)
        .with_filter_transition(crossfade(7))
        .build()
        .unwrap();

    for i in 0..300 {
        let filter = transform
            .prepare(&tap(1, i as f32 / 300.0, [0, 0]))
            .unwrap();
        drop(renderer.set_prepared_filter(filter).unwrap());

        for _ in 0..i % 5 {
            render(&mut renderer, [1.0; 4]);
        }
    }

    // Finish the active fade and its queued successor without another installation.
    render(&mut renderer, [1.0; 16]);

    let mut retired = 0;

    while renderer.take_retired_filter().is_some() {
        retired += 1;
    }

    assert!(retired <= 2);
}

#[test]
fn target_delay_is_primed_before_fading() {
    let mut renderer = renderer_fixture()
        .with_filter_transition(crossfade(8))
        .with_max_delay(seconds(12))
        .build()
        .unwrap();

    renderer.set_filter(&tap(1, 1.0, [0, 0])).unwrap();
    render(&mut renderer, [1.0; 12]);

    renderer.set_filter(&tap(1, 0.5, [5, 9])).unwrap();
    let [left, right] = render(&mut renderer, [1.0; 20]);

    // The slower ear needs nine warmup samples before either channel starts fading.
    let expected: Vec<_> = (0..20)
        .map(|i| {
            if i < 9 {
                1.0
            } else {
                let (out, input) = FadeCurve::Linear.gains(i - 9, 8);
                out + input * 0.5
            }
        })
        .collect();

    assert_close(&left, &expected);
    assert_close(&right, &expected);
}

#[test]
fn reset_restarts_warmup_and_preserves_queued_filter() {
    let fixture = || {
        renderer_fixture()
            .with_filter_transition(crossfade(8))
            .with_max_delay(seconds(12))
            .build()
            .unwrap()
    };

    // Reset once during warmup and once after the fade has begun.
    for partitions_before_reset in [1, 2] {
        let mut renderer = fixture();
        renderer.set_filter(&tap(1, 0.5, [5, 9])).unwrap();
        render(&mut renderer, [1.0; 12]);

        renderer.set_filter(&tap(1, 0.25, [3, 6])).unwrap();

        for _ in 0..partitions_before_reset {
            render(&mut renderer, [1.0; 4]);
        }

        renderer.set_filter(&tap(1, 0.75, [12, 10])).unwrap();
        renderer.reset();

        let mut fresh = fixture();

        for (gain, delays) in [(0.5, [5, 9]), (0.25, [3, 6]), (0.75, [12, 10])] {
            fresh.set_filter(&tap(1, gain, delays)).unwrap();
        }

        let [left, right] = render(&mut renderer, [1.0; 40]);
        let [expected_left, expected_right] = render(&mut fresh, [1.0; 40]);

        assert_close(&left, &expected_left);
        assert_close(&right, &expected_right);

        // Observe the queued filter after both fades, not just its silent warmup.
        assert_close(&left[36..], &[0.75; 4]);
        assert_close(&right[36..], &[0.75; 4]);
    }
}

#[test]
fn additive_output_matches_overwrite_during_delayed_queued_transitions() {
    let mut renderer = renderer_fixture()
        .with_filter_transition(crossfade(5))
        .with_max_delay(seconds(8))
        .build()
        .unwrap();

    for (gain, delays) in [(1.0, [0, 0]), (0.5, [8, 3]), (0.25, [0, 4])] {
        renderer.set_filter(&tap(1, gain, delays)).unwrap();
    }

    let mut additive = renderer.clone();
    let input: [f32; 40] = std::array::from_fn(|i| (i as f32 * 0.7).sin());
    let [mut left, mut right] = render(&mut renderer, input);

    let mut bus_left = [10.0; 40];
    let mut bus_right = [-10.0; 40];

    additive
        .process_block_add(&input, &mut bus_left, &mut bus_right)
        .unwrap();

    for value in &mut left {
        *value += 10.0;
    }

    for value in &mut right {
        *value -= 10.0;
    }

    assert_close(&bus_left, &left);
    assert_close(&bus_right, &right);
}

#[test]
fn invalid_block_lengths_preserve_output_and_history() {
    let mut renderer = renderer_fixture().build().unwrap();
    renderer.set_filter(&tap(1, 1.0, [0, 0])).unwrap();

    let mut left = [10.0; 3];
    let mut right = [20.0; 3];

    assert!(matches!(
        renderer.process_block(&[1.0; 3], &mut left, &mut right),
        Err(Error::InvalidInputOutputLen {
            actual: 3,
            partition_len: 4
        })
    ));

    assert!(
        renderer
            .process_block_add(&[1.0; 3], &mut left, &mut right)
            .is_err()
    );

    assert_eq!(left, [10.0; 3]);
    assert_eq!(right, [20.0; 3]);

    renderer.process_block(&[], &mut [], &mut []).unwrap();
}
