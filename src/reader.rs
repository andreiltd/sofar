//! This module provides high level bindings to [`libmysofa`] API allows to read
//! `HRTF` filters from `SOFA` files (Spatially Oriented Format for Acoustics).
//!
//! [`libmysofa`]: https://github.com/hoene/libmysofa

use std::{ffi::CString, io, path::Path};

const DEFAULT_CACHED: bool = false;
const DEFAULT_NORMALIZED: bool = true;

const DEFAULT_SAMPLE_RATE: f32 = 48000.0;

const DEFAULT_NEIGHBOR_ANGLE_STEP: f32 = ffi::MYSOFA_DEFAULT_NEIGH_STEP_ANGLE as f32;
const DEFAULT_NEIGHBOR_RADIUS_STEP: f32 = ffi::MYSOFA_DEFAULT_NEIGH_STEP_RADIUS as f32;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("The owls are not what they seem")]
    InternalError,
    #[error("Invalid data format")]
    InvalidFormat,
    #[error("Format is not supported")]
    UnsupportedFormat,
    #[error("Invalid attributes")]
    InvalidAttributes,
    #[error("Invalid dimensions")]
    InvalidDimensions,
    #[error("Invalid dimension list")]
    InvalidDimensionList,
    #[error("Invalid coordinate type")]
    InvalidCoordinateType,
    #[error("Invalid receiver position")]
    InvalidReceiverPositions,
    #[error("Emitters without ECI are not supported")]
    OnlyEmitterWithEciSupported,
    #[error("Delays without IR or MR are not supported")]
    OnlyDelaysWithIrOrMrSupported,
    #[error("Sources without MC are not supported")]
    OnlySourcesWithMcSupported,
    #[error("Sampling rates differ")]
    OnlyTheSameSamplingRateSupported,
}

impl Error {
    pub(crate) fn from_raw(err: i32) -> Error {
        use Error::*;

        match err {
            ffi::MYSOFA_INVALID_FORMAT => InvalidFormat,
            ffi::MYSOFA_UNSUPPORTED_FORMAT => UnsupportedFormat,
            ffi::MYSOFA_INVALID_ATTRIBUTES => InvalidAttributes,
            ffi::MYSOFA_INVALID_DIMENSIONS => InvalidDimensions,
            ffi::MYSOFA_INVALID_DIMENSION_LIST => InvalidDimensionList,
            ffi::MYSOFA_INVALID_COORDINATE_TYPE => InvalidCoordinateType,
            ffi::MYSOFA_INVALID_RECEIVER_POSITIONS => InvalidReceiverPositions,
            ffi::MYSOFA_ONLY_EMITTER_WITH_ECI_SUPPORTED => OnlyEmitterWithEciSupported,
            ffi::MYSOFA_ONLY_DELAYS_WITH_IR_OR_MR_SUPPORTED => OnlyDelaysWithIrOrMrSupported,
            ffi::MYSOFA_ONLY_SOURCES_WITH_MC_SUPPORTED => OnlySourcesWithMcSupported,
            ffi::MYSOFA_ONLY_THE_SAME_SAMPLING_RATE_SUPPORTED => OnlyTheSameSamplingRateSupported,
            ffi::MYSOFA_READ_ERROR => Io(io::Error::new(
                io::ErrorKind::NotFound,
                "Unable to read from file",
            )),
            ffi::MYSOFA_NO_MEMORY => Io(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Ran out of memory",
            )),
            _ => Error::InternalError,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    sample_rate: f32,
    neighbor_angle_step: f32,
    neighbor_radius_step: f32,
    cached: bool,
    normalized: bool,
}

impl OpenOptions {
    pub fn new() -> Self {
        Default::default()
    }

    /// Set sampling rate of HRTF data. If requested sampling rate is different
    /// than what is in a SOFA file, the data will be resampled. Default value
    /// is 48_000.0.
    pub fn sample_rate(&mut self, sample_rate: f32) -> &mut Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Neighbor search angle step measured in degrees. Default value is 0.5.
    ///
    /// The higher the value the faster search algorithm. The tradeoff
    /// is accuracy: higher values will more likely miss a true nearest
    /// neighbors.
    pub fn neighbor_angle_step(&mut self, neighbor_angle_step: f32) -> &mut Self {
        self.neighbor_angle_step = neighbor_angle_step;
        self
    }

    /// Neighbor search radius step measured in meters. Default value is 0.01.
    ///
    /// The higher the value the faster search algorithm. The tradeoff
    /// is accuracy: higher values will more likely miss a true nearest
    /// neighbors.
    pub fn neighbor_radius_step(&mut self, neighbor_radius_step: f32) -> &mut Self {
        self.neighbor_radius_step = neighbor_radius_step;
        self
    }

    /// Using this option tells library to share memory for the files with the
    /// same name and sampling rate.
    pub fn cached(&mut self, cached: bool) -> &mut Self {
        self.cached = cached;
        self
    }

    /// Apply normalization upon opening a SOFA file. Default value is `true`
    pub fn normalized(&mut self, normalized: bool) -> &mut Self {
        self.normalized = normalized;
        self
    }

    /// Open a SOFA file at `path` with open options specified in `self`
    ///
    /// ```no_run
    /// use sofar::reader::OpenOptions;
    ///
    /// let sofa = OpenOptions::new()
    ///     .normalized(false)
    ///     .sample_rate(44100.0)
    ///     .open("my/sofa/file.sofa")
    ///     .unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(&self, path: P) -> Result<Sofar, Error> {
        let path = cstr(path.as_ref())?;
        let mut filter_len = 0;
        let mut err = 0;

        let raw = unsafe {
            match self.cached {
                true => ffi::mysofa_open_cached(
                    path.as_ptr(),
                    self.sample_rate,
                    &mut filter_len,
                    &mut err,
                ),
                false => ffi::mysofa_open_advanced(
                    path.as_ptr(),
                    self.sample_rate,
                    &mut filter_len,
                    &mut err,
                    self.normalized,
                    self.neighbor_angle_step,
                    self.neighbor_radius_step,
                ),
            }
        };

        if raw.is_null() || err != ffi::MYSOFA_OK {
            return Err(Error::from_raw(err));
        }

        Ok(Sofar {
            raw,
            filter_len: filter_len as usize,
            cached: self.cached,
        })
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            sample_rate: DEFAULT_SAMPLE_RATE,
            neighbor_angle_step: DEFAULT_NEIGHBOR_ANGLE_STEP,
            neighbor_radius_step: DEFAULT_NEIGHBOR_RADIUS_STEP,
            cached: DEFAULT_CACHED,
            normalized: DEFAULT_NORMALIZED,
        }
    }
}

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

pub struct Sofar {
    raw: *mut ffi::MYSOFA_EASY,
    filter_len: usize,
    cached: bool,
}

impl Sofar {
    /// Open a SOFA file with the default open options
    ///
    /// ```no_run
    /// use sofar::reader::Sofar;
    ///
    /// let sofa = Sofar::open("my/sofa/file.sofa").unwrap();
    /// ```
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Sofar, Error> {
        OpenOptions::new().open(path)
    }

    pub fn filter_len(&self) -> usize {
        self.filter_len
    }

    /// Get HRTF filter for a given position
    ///
    /// To produce a stereo output for a given position a source should be
    /// delayed by left and right delay and FIR filtered by left and right
    /// impulse response.
    ///
    /// ```no_run
    /// use sofar::reader::{Sofar, Filter};
    ///
    /// let sofa = Sofar::open("my/sofa/file.sofa").unwrap();
    /// let filt_len = sofa.filter_len();
    ///
    /// let mut filter = Filter::new(filt_len);
    ///
    /// sofa.filter(0.0, 1.0, 0.0, &mut filter);
    /// ```
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `filter.left.len() < self.filter_len`
    /// - `filter.right.len() < self.filter_len`
    pub fn filter(&self, x: f32, y: f32, z: f32, filter: &mut Filter) {
        assert!(filter.left.len() >= self.filter_len);
        assert!(filter.right.len() >= self.filter_len);

        unsafe {
            ffi::mysofa_getfilter_float(
                self.raw,
                x,
                y,
                z,
                filter.left.as_mut_ptr(),
                filter.right.as_mut_ptr(),
                &mut filter.ldelay,
                &mut filter.rdelay,
            );
        }
    }

    /// Get HRTF filter for a given position with no interpolation
    ///
    /// Similar to [`filter`](crate::reader::Filter) method but it will skip the linear
    /// interpolation and return the filter for the nearest position instead.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - `filter.left.len() < self.filter_len`
    /// - `filter.right.len() < self.filter_len`
    pub fn filter_nointerp(&self, x: f32, y: f32, z: f32, filter: &mut Filter) {
        assert!(filter.left.len() >= self.filter_len);
        assert!(filter.right.len() >= self.filter_len);

        unsafe {
            ffi::mysofa_getfilter_float_nointerp(
                self.raw,
                x,
                y,
                z,
                filter.left.as_mut_ptr(),
                filter.right.as_mut_ptr(),
                &mut filter.ldelay,
                &mut filter.rdelay,
            );
        }
    }
}

impl Drop for Sofar {
    fn drop(&mut self) {
        unsafe {
            match self.cached {
                true => ffi::mysofa_close_cached(self.raw),
                false => ffi::mysofa_close(self.raw),
            }
        }
    }
}

unsafe impl Send for Sofar {}
unsafe impl Sync for Sofar {}

#[cfg(unix)]
fn cstr(path: &Path) -> std::io::Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    Ok(CString::new(path.as_os_str().as_bytes())?)
}

#[cfg(windows)]
fn cstr(path: &Path) -> std::io::Result<CString> {
    use std::os::windows::ffi::OsStrExt;
    Ok(CString::new(path.as_os_str().as_bytes())?)
}
