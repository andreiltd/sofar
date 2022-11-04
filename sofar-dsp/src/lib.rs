use std::sync::Arc;

use sofar::Filter;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
const DEFAULT_BLOCK_LEN: usize = 256;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Sample rate is invalid: {0}")]
    InvalidSampleRate(f32),
    #[error("Filter length ({0}) is invalid: {1}")]
    InvalidFilterLength(usize, usize),
    #[error("Input/Output length ({0}) should be multiple of block len ({1})")]
    InvalidInputOutputLen(usize, usize),
    #[error("The owls are not what they seem")]
    InternalProcessingError(#[from] realfft::FftError),
}

struct Delay {
    buf: Vec<f32>,
    delay: usize,
    rpos: usize,
    wpos: usize,
}

impl Delay {
    fn new(delay: usize) -> Self {
        Self {
            buf: vec![0.0; delay + 1],
            delay,
            rpos: 1,
            wpos: 0,
        }
    }

    fn set_delay(&mut self, delay: usize) {
        let n = self.buf.len();

        if delay >= n {
            self.buf.resize(delay + 1, 0.0)
        }

        if self.wpos >= delay {
            self.rpos = self.wpos - delay;
        } else {
            self.rpos = n + self.wpos - delay;
        }

        self.delay = delay;
    }

    fn next(&mut self, input: f32) -> f32 {
        self.buf[self.wpos] = input;
        self.wpos = (self.wpos + 1) % self.buf.len();

        let output = self.buf[self.rpos];
        self.rpos = (self.rpos + 1) % self.buf.len();

        output
    }

    fn apply(&mut self, buf: &mut [f32]) {
        for sample in buf {
            *sample = self.next(*sample);
        }
    }
}

struct Channel {
    /// impulse response split into partition blocks
    h: Box<[Complex<f32>]>,
    /// input blocks frequency domain delay line
    x_fdl: Box<[Complex<f32>]>,
    /// input blocks time domain delay line
    x_tdl: Box<[f32]>,
    /// left channel delay state
    delay: Option<Delay>,
}

impl Channel {
    fn new(
        fft_len: usize,
        spectra_len: usize,
        partitions: usize,
        sample_rate: f32,
        delay: Option<f32>,
    ) -> Self {
        let zero = Complex::new(0.0, 0.0);

        let h = vec![zero; spectra_len * partitions].into_boxed_slice();
        let x_fdl = vec![zero; spectra_len * partitions].into_boxed_slice();
        let x_tdl = vec![0.0; fft_len].into_boxed_slice();

        let delay = delay.and_then(|delay| {
            if delay > 0.0 {
                Some(Delay::new((delay * sample_rate) as usize))
            } else {
                None
            }
        });

        Channel {
            h,
            x_tdl,
            x_fdl,
            delay,
        }
    }

    fn delay<O>(&mut self, mut buf: O)
    where
        O: AsMut<[f32]>,
    {
        if let Some(delay) = self.delay.as_mut() {
            delay.apply(buf.as_mut());
        }
    }

    fn update_delay(&mut self, new_delay: usize) {
        if let Some(delay) = self.delay.as_mut() {
            if new_delay != delay.delay {
                delay.set_delay(new_delay)
            }
        }
    }
}

#[must_use]
pub struct RendererBuilder {
    sample_rate: f32,
    kernel_len: usize,
    block_len: usize,
    left_delay: Option<f32>,
    right_delay: Option<f32>,
}

impl RendererBuilder {
    fn new(kernel_len: usize) -> RendererBuilder {
        RendererBuilder {
            kernel_len,
            sample_rate: DEFAULT_SAMPLE_RATE,
            block_len: DEFAULT_BLOCK_LEN,
            left_delay: None,
            right_delay: None,
        }
    }

    pub fn with_sample_rate(mut self, sample_rate: f32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_block_len(mut self, block_len: usize) -> Self {
        self.block_len = block_len;
        self
    }

    pub fn with_left_delay(mut self, left_delay: f32) -> Self {
        self.left_delay = Some(left_delay);
        self
    }

    pub fn with_right_delay(mut self, right_delay: f32) -> Self {
        self.right_delay = Some(right_delay);
        self
    }

    pub fn build(self) -> Result<Renderer, Error> {
        let sample_rate = match self.sample_rate.is_normal() && self.sample_rate.is_sign_positive()
        {
            true => self.sample_rate,
            false => return Err(Error::InvalidSampleRate(self.sample_rate)),
        };

        let partitions = (self.kernel_len + self.block_len - 1) / self.block_len;

        let fft_len = self.block_len * 2;
        let spectra_len = fft_len / 2 + 1;
        let zero = Complex::new(0.0, 0.0);

        let scratch = vec![0.0; fft_len].into_boxed_slice();
        let filt_pad = vec![0.0; fft_len].into_boxed_slice();
        let acc = vec![zero; spectra_len].into_boxed_slice();

        let mut planner = RealFftPlanner::<f32>::new();
        let rfft = planner.plan_fft_forward(fft_len);
        let ifft = planner.plan_fft_inverse(fft_len);

        let rfft_scratch = rfft.make_scratch_vec();
        let ifft_scratch = ifft.make_scratch_vec();

        let left = Channel::new(
            fft_len,
            spectra_len,
            partitions,
            sample_rate,
            self.left_delay,
        );
        let right = Channel::new(
            fft_len,
            spectra_len,
            partitions,
            sample_rate,
            self.right_delay,
        );

        let state = State {
            acc,
            rfft,
            ifft,
            fft_len,
            scratch,
            filt_pad,
            rfft_scratch,
            ifft_scratch,
            partitions,
            sample_rate: self.sample_rate,
            kernel_len: self.kernel_len,
            block_len: self.block_len,
        };

        Ok(Renderer { left, right, state })
    }
}

pub struct Renderer {
    /// common state
    state: State,
    /// left channel data
    left: Channel,
    /// right channel data
    right: Channel,
}

impl Renderer {
    pub fn builder(kernel_len: usize) -> RendererBuilder {
        RendererBuilder::new(kernel_len)
    }

    /// # Panics
    ///
    /// This method panics if:
    /// - `filt.left.len() != filt.right.len())`
    pub fn update_filter(&mut self, filt: &Filter) -> Result<(), Error> {
        assert_eq!(filt.left.len(), filt.right.len());

        if self.state.kernel_len != filt.left.len() {
            return Err(Error::InvalidFilterLength(
                filt.left.len(),
                self.state.kernel_len,
            ));
        }

        self.state.filt_split(&filt.left, &mut self.left.h)?;
        self.state.filt_split(&filt.right, &mut self.right.h)?;

        self.left
            .update_delay((filt.ldelay * self.state.sample_rate) as usize);
        self.right
            .update_delay((filt.rdelay * self.state.sample_rate) as usize);

        Ok(())
    }

    /// # Panics
    ///
    /// This method panics if:
    /// - `input.len() != left.len()`
    /// - `input.len() != right.len()`
    pub fn process_block<I: AsRef<[f32]>, O: AsMut<[f32]>>(
        &mut self,
        input: I,
        mut left: O,
        mut right: O,
    ) -> Result<(), Error> {
        assert_eq!(left.as_mut().len(), input.as_ref().len());
        assert_eq!(right.as_mut().len(), input.as_ref().len());

        if usize::rem_euclid(left.as_mut().len(), self.state.block_len) != 0 {
            return Err(Error::InvalidInputOutputLen(
                left.as_mut().len(),
                self.state.block_len,
            ));
        }

        self.state
            .conv(&mut self.left, input.as_ref(), left.as_mut())?;
        self.state
            .conv(&mut self.right, input.as_ref(), right.as_mut())?;

        self.left.delay(left.as_mut());
        self.right.delay(right.as_mut());

        Ok(())
    }
}

struct State {
    /// Sample rate
    sample_rate: f32,
    /// Length of the filter
    kernel_len: usize,
    /// Length of the processing block in samples
    block_len: usize,
    /// Number of the partitions for uniformly partitioned convolution
    partitions: usize,
    /// FFT size
    fft_len: usize,
    /// Real FFT module
    rfft: Arc<dyn RealToComplex<f32>>,
    /// Inverse FFT module
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// RFFT scratch memory
    rfft_scratch: Vec<Complex<f32>>,
    /// RFFT scratch memory
    ifft_scratch: Vec<Complex<f32>>,
    /// mutable internal scratch for fft input
    scratch: Box<[f32]>,
    /// filter padding to block_size * 2
    filt_pad: Box<[f32]>,
    /// accumulator for point wise multiplication
    acc: Box<[Complex<f32>]>,
}

impl State {
    fn conv<I, O>(&mut self, channel: &mut Channel, x: I, mut y: O) -> Result<(), Error>
    where
        I: AsRef<[f32]>,
        O: AsMut<[f32]>,
    {
        let x = x.as_ref();
        let y = y.as_mut();

        let spectra_len = self.fft_len / 2 + 1;
        let block_len = self.block_len;
        let scale = self.fft_len as f32;

        let mut off = 0;

        while off < x.len() {
            // shift right part of the buffer to the left
            channel.x_tdl.copy_within(block_len.., 0);
            // store new data in right part
            channel.x_tdl[block_len..].copy_from_slice(&x[off..off + block_len]);
            // shift up the fdl content by one slot
            channel.x_fdl.rotate_right(spectra_len);
            // move data to processing scratch
            self.scratch.copy_from_slice(&channel.x_tdl);
            // take real to complex fft of input block and store it in the first fdl slot
            self.rfft.process_with_scratch(
                &mut self.scratch,
                &mut channel.x_fdl[..spectra_len],
                &mut self.rfft_scratch,
            )?;

            // point wise multiply with filter and accumulate the results
            let mut p_off = 0;
            self.acc.fill(Complex::new(0.0, 0.0));

            for _ in 0..self.partitions {
                for (acc, (x, h)) in Iterator::zip(
                    self.acc.iter_mut(),
                    Iterator::zip(
                        channel.x_fdl[p_off..p_off + spectra_len].iter(),
                        channel.h[p_off..p_off + spectra_len].iter(),
                    ),
                ) {
                    *acc += x * h;
                }

                p_off += spectra_len;
            }

            // take complex to real transform
            self.ifft.process_with_scratch(
                &mut self.acc,
                &mut self.scratch,
                &mut self.ifft_scratch,
            )?;

            // discard the left part and write the right part as the next output block
            for (y, x) in Iterator::zip(
                y[off..off + block_len].iter_mut(),
                self.scratch[block_len..].iter(),
            ) {
                *y = x / scale;
            }

            // update offset
            off += block_len;
        }

        Ok(())
    }

    fn filt_split(&mut self, taps: &[f32], h: &mut [Complex<f32>]) -> Result<(), Error> {
        assert!(taps.len() <= h.len());

        let spectra_len = self.fft_len / 2 + 1;
        let block_len = self.block_len;

        let mut off = 0;
        let mut iter = taps.chunks_exact(block_len);

        for partition in iter.by_ref() {
            self.filt_pad[..block_len].copy_from_slice(partition);

            self.rfft.process_with_scratch(
                &mut self.filt_pad,
                &mut h[off..off + spectra_len],
                &mut self.rfft_scratch,
            )?;

            off += spectra_len;
        }

        let remainder = iter.remainder();
        let remainder_len = remainder.len();

        if remainder_len > 0 {
            self.scratch[..remainder_len].copy_from_slice(remainder);
            self.scratch[remainder_len..].fill(0.0);

            self.rfft.process_with_scratch(
                &mut self.scratch,
                &mut h[off..off + spectra_len],
                &mut self.rfft_scratch,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn convolve_from_definition(x: Vec<f32>, h: Vec<f32>) -> Vec<f32> {
        let mut x_neg_terms = vec![0.0; h.len() - 1];
        x_neg_terms.extend(x.clone());

        (0..x.len())
            .map(|i| {
                Iterator::zip(x_neg_terms.iter().skip(i), h.iter().rev())
                    .map(|(x, h)| x * h)
                    .fold(0.0, |acc, x| acc + x)
            })
            .collect::<Vec<_>>()
    }

    #[must_use]
    struct ConvTest {
        kernel_len: usize,
        block_len: usize,
        input_len: usize,
    }

    impl Default for ConvTest {
        fn default() -> Self {
            Self {
                kernel_len: 256,
                block_len: 64,
                input_len: 128,
            }
        }
    }

    impl ConvTest {
        fn kernel_len(mut self, kernel_len: usize) -> Self {
            self.kernel_len = kernel_len;
            self
        }

        fn block_len(mut self, block_len: usize) -> Self {
            self.block_len = block_len;
            self
        }

        fn input_len(mut self, input_len: usize) -> Self {
            self.input_len = input_len;
            self
        }

        fn run(&self) {
            let mut renderer = Renderer::builder(self.kernel_len)
                .with_block_len(self.block_len)
                .build()
                .expect("renderer");

            let input = (1..=self.input_len).map(|v| v as f32).collect::<Vec<_>>();

            let h = (1..=self.kernel_len)
                .map(|v| v as f32)
                .collect::<Vec<_>>()
                .into_boxed_slice();

            let mut left = vec![0.0; self.input_len];
            let mut right = vec![0.0; self.input_len];

            let filt = Filter {
                left: h.clone(),
                right: h.clone(),
                ldelay: 0.0,
                rdelay: 0.0,
            };

            renderer.update_filter(&filt).expect("filter updated");

            renderer
                .process_block(&input, &mut left, &mut right)
                .expect("render block");

            let expected = convolve_from_definition(input.to_vec(), h.to_vec());

            for (a, b) in std::iter::zip(expected.iter(), left.iter()) {
                assert_approx_eq!(a, b, 0.5);
            }

            for (a, b) in std::iter::zip(expected.iter(), right.iter()) {
                assert_approx_eq!(a, b, 0.5);
            }
        }
    }

    #[test]
    fn conv_default() {
        ConvTest::default().run();
    }

    #[test]
    fn conv_long_kernel() {
        ConvTest::default()
            .kernel_len(4096)
            .block_len(64)
            .input_len(256)
            .run();
    }

    #[test]
    fn conv_short_kernel() {
        ConvTest::default()
            .kernel_len(16)
            .block_len(4)
            .input_len(256)
            .run();
    }

    #[test]
    fn conv_kernel_and_block_same_length() {
        ConvTest::default()
            .kernel_len(16)
            .block_len(16)
            .input_len(96)
            .run();
    }

    #[test]
    fn conv_odd_kernel() {
        ConvTest::default()
            .kernel_len(1025)
            .block_len(16)
            .input_len(256)
            .run();
    }

    #[test]
    fn conv_even_kernel() {
        ConvTest::default()
            .kernel_len(100)
            .block_len(32)
            .input_len(32)
            .run();
    }

    #[test]
    fn delay() {
        let mut delay = Delay::new(42);
        let mut input = (1..=128).map(|v| v as f32).collect::<Vec<_>>();
        let mut expected = input.clone();

        expected.rotate_right(42);

        for i in 0..42 {
            expected[i] = 0.0;
        }

        delay.apply(input.as_mut_slice());
        assert_eq!(input, expected);
    }
}
