/// HRTF filter output containing left and right channel impulse responses
/// and interaural time difference (ITD) delays.
pub struct Filter {
    /// Impulse Response of FIR filter for left channel
    pub left: Box<[f32]>,
    /// Impulse Response of FIR filter for right channel
    pub right: Box<[f32]>,
    /// The amount of time in seconds that left channel should be delayed for
    pub ldelay: f32,
    /// The amount of time in seconds that right channel should be delayed for
    pub rdelay: f32,
}

impl Filter {
    pub fn new(filt_len: usize) -> Self {
        Self {
            left: vec![0.0; filt_len].into_boxed_slice(),
            right: vec![0.0; filt_len].into_boxed_slice(),
            ldelay: 0.0,
            rdelay: 0.0,
        }
    }
}
