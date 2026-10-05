use super::{assert_close, convolve, plan, seconds, tap};
use crate::render::Renderer;

/// Numerical convolution case with deterministic input and an independent oracle.
///
/// Defaults to a 256-tap filter, 64-sample partitions, and 512 input samples.
/// One-tap cases are passthrough filters; longer filters have distinct stereo tails.
#[must_use]
struct ConvTest {
    /// FIR taps per ear.
    filter_len: usize,
    /// Samples processed in each call, preserving history between calls.
    partition_len: usize,
    /// Total input samples, a multiple of the partition length.
    input_len: usize,
    /// Left/right filter delays in samples, even when delay processing is disabled.
    filter_delays: [usize; 2],
    /// Inclusive per-ear delay limit, or `None` to ignore filter delays.
    delay_capacity: Option<usize>,
    /// Initial left/right bus levels, or `None` for overwrite processing.
    additive_bus: Option<[f32; 2]>,
}

impl Default for ConvTest {
    fn default() -> Self {
        Self {
            filter_len: 256,
            partition_len: 64,
            input_len: 512,
            filter_delays: [0, 0],
            delay_capacity: None,
            additive_bus: None,
        }
    }
}

impl ConvTest {
    /// Set the FIR tap count per ear.
    fn filter_len(mut self, filter_len: usize) -> Self {
        self.filter_len = filter_len;

        self
    }

    /// Set the number of samples processed per call.
    fn partition_len(mut self, partition_len: usize) -> Self {
        self.partition_len = partition_len;

        self
    }

    /// Set the total input length in samples.
    fn input_len(mut self, input_len: usize) -> Self {
        self.input_len = input_len;

        self
    }

    /// Set delay metadata without implicitly enabling renderer delays.
    fn filter_delays(mut self, delays: [usize; 2]) -> Self {
        self.filter_delays = delays;

        self
    }

    /// Enable filter delays with an inclusive per-ear capacity.
    fn delay_capacity(mut self, max_samples: usize) -> Self {
        self.delay_capacity = Some(max_samples);

        self
    }

    /// Exercise additive rendering into a bus with the supplied channel levels.
    fn add_to_bus(mut self, levels: [f32; 2]) -> Self {
        self.additive_bus = Some(levels);

        self
    }

    /// Render consecutive partitions and compare both ears with direct convolution.
    ///
    /// Build the fixture, process without resetting history, then apply delays and
    /// initial bus levels to the independent time-domain reference.
    fn run(self) {
        let plan = plan(self.filter_len, self.partition_len);
        let mut builder = Renderer::builder(&plan);

        if let Some(max_samples) = self.delay_capacity {
            builder = builder.with_max_delay(seconds(max_samples));
        }

        let mut renderer = builder.build().unwrap();
        let mut filter = tap(self.filter_len, 1.0, self.filter_delays);

        for (i, (left, right)) in filter
            .left
            .iter_mut()
            .zip(&mut filter.right)
            .enumerate()
            .skip(1)
        {
            *left = (i % 17) as f32 / 17.0 - 0.5;
            *right = (i % 11) as f32 / 11.0 - 0.5;
        }

        renderer.set_filter(&filter).unwrap();

        let input: Vec<_> = (0..self.input_len)
            .map(|i| (i % 13) as f32 / 13.0 - 0.5)
            .collect();
        let bus = self.additive_bus.unwrap_or([0.0; 2]);
        let mut left = vec![bus[0]; self.input_len];
        let mut right = vec![bus[1]; self.input_len];

        for ((input, left), right) in input
            .chunks(self.partition_len)
            .zip(left.chunks_mut(self.partition_len))
            .zip(right.chunks_mut(self.partition_len))
        {
            if self.additive_bus.is_some() {
                renderer.process_block_add(input, left, right).unwrap();
            } else {
                renderer.process_block(input, left, right).unwrap();
            }
        }

        let expected_delays = match self.delay_capacity {
            Some(_) => self.filter_delays,
            None => [0, 0],
        };
        let taps = [&filter.left, &filter.right];

        for (ear, actual) in [left, right].iter().enumerate() {
            let mut expected = convolve(&input, taps[ear], expected_delays[ear]);

            for sample in &mut expected {
                *sample += bus[ear];
            }

            assert_close(actual, &expected);
        }
    }
}

#[test]
fn conv_default() {
    ConvTest::default().run();
}

#[test]
fn conv_long_kernel() {
    ConvTest::default().filter_len(4096).input_len(256).run();
}

#[test]
fn conv_short_kernel() {
    ConvTest::default()
        .filter_len(16)
        .partition_len(4)
        .input_len(256)
        .run();
}

#[test]
fn conv_kernel_and_partition_have_equal_lengths() {
    ConvTest::default()
        .filter_len(16)
        .partition_len(16)
        .input_len(96)
        .run();
}

#[test]
fn conv_odd_kernel() {
    ConvTest::default()
        .filter_len(1025)
        .partition_len(16)
        .input_len(1200)
        .run();
}

#[test]
fn conv_even_kernel() {
    ConvTest::default()
        .filter_len(100)
        .partition_len(32)
        .input_len(320)
        .run();
}

#[test]
fn conv_non_power_of_two_partition() {
    ConvTest::default()
        .filter_len(1)
        .partition_len(7)
        .input_len(70)
        .run();
}

#[test]
fn delay_capacity_is_independent_of_fir_length() {
    ConvTest::default()
        .filter_len(1)
        .partition_len(4)
        .input_len(24)
        .filter_delays([8, 12])
        .delay_capacity(12)
        .run();
}

#[test]
fn disabled_delays_ignore_filter_metadata() {
    ConvTest::default()
        .filter_len(1)
        .partition_len(4)
        .input_len(24)
        .filter_delays([8, 12])
        .run();
}

#[test]
fn zero_delay_capacity_is_passthrough() {
    ConvTest::default()
        .filter_len(1)
        .partition_len(4)
        .input_len(4)
        .delay_capacity(0)
        .run();
}

#[test]
fn additive_output_preserves_the_existing_bus() {
    ConvTest::default().add_to_bus([10.0, -10.0]).run();
}

#[test]
fn additive_output_applies_filter_delays() {
    ConvTest::default()
        .filter_len(1)
        .partition_len(4)
        .input_len(24)
        .filter_delays([8, 12])
        .delay_capacity(12)
        .add_to_bus([10.0, -10.0])
        .run();
}
