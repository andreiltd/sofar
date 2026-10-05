#![cfg(feature = "dsp")]

use quickcheck::QuickCheck;
use sofar::{
    filter::Filter,
    render::{FadeCurve, FilterTransition, Renderer, RendererPlan},
};

const CASES: u64 = 100;

fn bounded(hint: u8, upper: usize) -> usize {
    usize::from(hint) % upper + 1
}

/// Generate repeatable, bounded samples from QuickCheck's seed.
fn samples(len: usize, mut state: u64) -> Vec<f32> {
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);

            let value = (state >> 48) as u16;
            value as f32 / u16::MAX as f32 * 2.0 - 1.0
        })
        .collect()
}

fn filter(filter_len: usize, seed: u64) -> Filter {
    Filter {
        left: samples(filter_len, seed ^ 0xa076_1d64_78bd_642f).into_boxed_slice(),
        right: samples(filter_len, seed ^ 0xe703_7ed1_a0b4_28db).into_boxed_slice(),
        ldelay: 0.0,
        rdelay: 0.0,
    }
}

fn single_tap_filter(gain: f32) -> Filter {
    Filter {
        left: vec![gain].into_boxed_slice(),
        right: vec![gain].into_boxed_slice(),
        ldelay: 0.0,
        rdelay: 0.0,
    }
}

fn close(actual: f32, expected: f32) -> bool {
    let scale = 1.0 + actual.abs().max(expected.abs());

    (actual - expected).abs() <= 1e-4 * scale
}

fn slices_close(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&actual, &expected)| close(actual, expected))
}

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

fn direct_and_planned_renderers_match(
    filter_hint: u8,
    partition_hint: u8,
    blocks_hint: u8,
    seed: u64,
) -> bool {
    let filter_len = bounded(filter_hint, 64);
    let partition_len = bounded(partition_hint, 32);
    let input_len = partition_len * bounded(blocks_hint, 4);
    let filter = filter(filter_len, seed);
    let input = samples(input_len, seed ^ 0x8ebc_6af0_9c88_c6e3);

    let direct_plan = RendererPlan::builder(filter_len)
        .with_partition_len(partition_len)
        .build()
        .expect("independent plan");

    let mut direct = Renderer::new(&direct_plan);

    direct.set_filter(&filter).expect("direct filter");

    let plan = RendererPlan::builder(filter_len)
        .with_partition_len(partition_len)
        .build()
        .expect("renderer plan");

    let mut transform = plan.filter_transform();
    let prepared_filter = transform.prepare(&filter).expect("prepared filter");
    let mut planned = Renderer::new(&plan);

    drop(
        planned
            .set_prepared_filter(prepared_filter)
            .expect("planned filter"),
    );

    let mut direct_left = vec![0.0; input_len];
    let mut direct_right = vec![0.0; input_len];
    let mut planned_left = vec![0.0; input_len];
    let mut planned_right = vec![0.0; input_len];

    direct
        .process_block(&input, &mut direct_left, &mut direct_right)
        .expect("direct processing");

    planned
        .process_block(&input, &mut planned_left, &mut planned_right)
        .expect("planned processing");

    slices_close(&direct_left, &planned_left)
        && slices_close(&direct_right, &planned_right)
        && slices_close(&direct_left, &convolve(&input, &filter.left, 0))
        && slices_close(&direct_right, &convolve(&input, &filter.right, 0))
}

fn additive_processing_matches_overwrite_plus_bus(
    filter_hint: u8,
    partition_hint: u8,
    blocks_hint: u8,
    seed: u64,
) -> bool {
    let filter_len = bounded(filter_hint, 64);
    let partition_len = bounded(partition_hint, 32);
    let input_len = partition_len * bounded(blocks_hint, 4);
    let filter = filter(filter_len, seed);
    let input = samples(input_len, seed ^ 0x5899_65cc_7537_4cc3);

    let plan = RendererPlan::builder(filter_len)
        .with_partition_len(partition_len)
        .build()
        .expect("shared plan");

    let mut overwrite = Renderer::new(&plan);

    let mut additive = Renderer::new(&plan);

    overwrite.set_filter(&filter).expect("overwrite filter");
    additive.set_filter(&filter).expect("additive filter");

    let mut rendered_left = vec![0.0; input_len];
    let mut rendered_right = vec![0.0; input_len];

    overwrite
        .process_block(&input, &mut rendered_left, &mut rendered_right)
        .expect("overwrite processing");

    let bus_left = samples(input_len, seed ^ 0x1d8e_4e27_c47d_124f);
    let bus_right = samples(input_len, seed ^ 0xeb44_acca_b455_d165);
    let mut expected_left = bus_left.clone();
    let mut expected_right = bus_right.clone();

    for (expected, rendered) in expected_left.iter_mut().zip(&rendered_left) {
        *expected += rendered;
    }

    for (expected, rendered) in expected_right.iter_mut().zip(&rendered_right) {
        *expected += rendered;
    }

    let mut actual_left = bus_left;
    let mut actual_right = bus_right;

    additive
        .process_block_add(&input, &mut actual_left, &mut actual_right)
        .expect("additive processing");

    slices_close(&actual_left, &expected_left) && slices_close(&actual_right, &expected_right)
}

fn crossfade_matches_linear_interpolation(
    partition_hint: u8,
    duration_hint: u8,
    initial_gain: i8,
    target_gain: i8,
    seed: u64,
) -> bool {
    let partition_len = bounded(partition_hint, 32);
    let duration_samples = bounded(duration_hint, partition_len * 4);
    let input_len = (duration_samples.div_ceil(partition_len) + 1) * partition_len;
    let initial_gain = f32::from(initial_gain) / 64.0;
    let target_gain = f32::from(target_gain) / 64.0;
    let input = samples(input_len, seed ^ 0x4f1b_cdc6_76b6_c2ed);

    let plan = RendererPlan::builder(1)
        .with_partition_len(partition_len)
        .build()
        .expect("renderer plan");

    let mut renderer = Renderer::builder(&plan)
        .with_filter_transition(FilterTransition::Crossfade {
            duration_samples,
            curve: FadeCurve::Linear,
        })
        .build()
        .expect("crossfade renderer");

    renderer
        .set_filter(&single_tap_filter(initial_gain))
        .expect("initial filter");
    renderer
        .set_filter(&single_tap_filter(target_gain))
        .expect("target filter");

    let mut left = vec![0.0; input_len];
    let mut right = vec![0.0; input_len];

    renderer
        .process_block(&input, &mut left, &mut right)
        .expect("crossfade processing");

    let expected = input
        .iter()
        .enumerate()
        .map(|(position, &sample)| {
            let fade_in = if duration_samples <= 1 || position >= duration_samples {
                1.0
            } else {
                position as f32 / (duration_samples - 1) as f32
            };

            sample * (initial_gain * (1.0 - fade_in) + target_gain * fade_in)
        })
        .collect::<Vec<_>>();

    slices_close(&left, &expected) && slices_close(&right, &expected)
}

fn delayed_crossfade_matches_direct_convolution(
    filter_hint: u8,
    partition_hint: u8,
    duration_hint: u8,
    delay_hint: u8,
    seed: u64,
) -> bool {
    let filter_len = bounded(filter_hint, 16);
    let partition_len = bounded(partition_hint, 8);
    let duration = bounded(duration_hint, 32);
    let old_delays = [usize::from(delay_hint) % 17, usize::from(delay_hint) % 9];
    let new_delays = [usize::from(delay_hint) % 13, usize::from(delay_hint) % 25];

    // Fill the initial history, then leave enough input for target warmup and the full fade.
    let prefix = (filter_len + 24).div_ceil(partition_len) * partition_len;
    let warmup = new_delays[0].max(new_delays[1]);
    let suffix = (warmup + duration).div_ceil(partition_len) * partition_len + partition_len;

    let input = samples(prefix + suffix, seed);
    let mut old = filter(filter_len, seed ^ 0x98a6_5c8d);
    let mut new = filter(filter_len, seed ^ 0x5287_a3f1);
    old.ldelay = old_delays[0] as f32 / 1024.0;
    old.rdelay = old_delays[1] as f32 / 1024.0;
    new.ldelay = new_delays[0] as f32 / 1024.0;
    new.rdelay = new_delays[1] as f32 / 1024.0;

    let plan = RendererPlan::builder(filter_len)
        .with_sample_rate(1024.0)
        .with_partition_len(partition_len)
        .build()
        .unwrap();

    let mut renderer = Renderer::builder(&plan)
        .with_max_delay(24.0 / 1024.0)
        .with_filter_transition(FilterTransition::Crossfade {
            duration_samples: duration,
            curve: FadeCurve::Linear,
        })
        .build()
        .unwrap();

    renderer.set_filter(&old).unwrap();
    let mut left = vec![0.0; input.len()];
    let mut right = vec![0.0; input.len()];

    renderer
        .process_block(&input[..prefix], &mut left[..prefix], &mut right[..prefix])
        .unwrap();

    renderer.set_filter(&new).unwrap();
    renderer
        .process_block(&input[prefix..], &mut left[prefix..], &mut right[prefix..])
        .unwrap();

    // Blend two full-history convolutions only after the target delay lines are primed.
    let expected = |old: &[f32], new: &[f32], channel: usize| {
        let old = convolve(&input, old, old_delays[channel]);
        let new = convolve(&input, new, new_delays[channel]);

        old.iter()
            .zip(&new)
            .enumerate()
            .map(|(i, (old, new))| {
                let fade_in = if i < prefix + warmup {
                    0.0
                } else if duration == 1 {
                    1.0
                } else {
                    ((i - prefix - warmup) as f32 / (duration - 1) as f32).min(1.0)
                };

                old * (1.0 - fade_in) + new * fade_in
            })
            .collect::<Vec<_>>()
    };

    slices_close(&left, &expected(&old.left, &new.left, 0))
        && slices_close(&right, &expected(&old.right, &new.right, 1))
}

#[test]
fn planned_renderers_match_direct_renderers() {
    QuickCheck::new()
        .tests(CASES)
        .quickcheck(direct_and_planned_renderers_match as fn(u8, u8, u8, u64) -> bool);
}

#[test]
fn additive_processing_preserves_the_existing_bus() {
    QuickCheck::new()
        .tests(CASES)
        .quickcheck(additive_processing_matches_overwrite_plus_bus as fn(u8, u8, u8, u64) -> bool);
}

#[test]
fn crossfades_match_linear_interpolation_across_partitions() {
    QuickCheck::new()
        .tests(CASES)
        .quickcheck(crossfade_matches_linear_interpolation as fn(u8, u8, i8, i8, u64) -> bool);
}

#[test]
fn delayed_crossfades_match_direct_convolution_with_full_input_history() {
    QuickCheck::new().tests(CASES).quickcheck(
        delayed_crossfade_matches_direct_convolution as fn(u8, u8, u8, u8, u64) -> bool,
    );
}
