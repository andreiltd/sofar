//! # Sofar
//!
//! Pure Rust SOFA Reader and HRTF Renderer
//!
//! This crate provides a pure Rust implementation for reading `HRTF` filters
//! from `SOFA` files (Spatially Oriented Format for Acoustics).
//!
//! The [`render`] module implements uniformly partitioned convolution algorithm
//! for rendering HRTF filters.
//!
//! Based on the [`libmysofa`] C library by Christian Hoene / Symonics GmbH.
//!
//! [`libmysofa`]: https://github.com/hoene/libmysofa
//! [`render`]: `crate::render`
//!
//! # Example
//!
//! ```no_run
//!
//! use sofar::reader::{Filter, OpenOptions};
//! use sofar::render::{Renderer, RendererPlan};
//!
//! // Open sofa file, resample HRTF data if needed to 44_100
//! let sofa = OpenOptions::new()
//!     .sample_rate(44100.0)
//!     .open("my/sofa/file.sofa")
//!     .unwrap();
//!
//! let filt_len = sofa.filter_len();
//! let mut filter = Filter::new(filt_len);
//!
//! // Get filter at position
//! sofa.filter(0.0, 1.0, 0.0, &mut filter);
//!
//! let plan = RendererPlan::builder(filt_len)
//!     .with_sample_rate(sofa.sample_rate())
//!     .with_partition_len(64)
//!     .build()
//!     .unwrap();
//!
//! let mut render = Renderer::new(&plan);
//! render.set_filter(&filter).unwrap();
//!
//! let input = vec![0.0; 256];
//! let mut left = vec![0.0; 256];
//! let mut right = vec![0.0; 256];
//!
//! // read_input()
//!
//! render.process_block(&input, &mut left, &mut right).unwrap();
//! ```
//!
//! Use [`Renderer::builder`](render::Renderer::builder) to enable crossfades
//! or filter delays. For real-time filter updates, split a renderer with
//! [`into_realtime`](render::Renderer::into_realtime): a worker thread
//! publishes filters through the [`render::FilterPublisher`], and the
//! [`render::RealtimeRenderer`] adopts them in the audio callback without
//! allocating or freeing memory.

pub mod filter;

pub mod hdf;
pub mod reader;
mod sofa;

#[cfg(feature = "dsp")]
pub mod render;

#[cfg(test)]
mod tests {
    use super::*;
    use reader::{Filter, Sofar};

    #[test]
    fn open_test() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        std::env::set_current_dir(cwd).unwrap();

        let sofa = Sofar::open("libmysofa-sys/libmysofa/tests/tester.sofa").unwrap();
        let filt_len = sofa.filter_len();

        let mut filter = Filter::new(filt_len);
        sofa.filter(0.0, 1.0, 0.0, &mut filter);
    }

    #[test]
    fn open_data_test() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        std::env::set_current_dir(cwd).unwrap();

        let data = std::fs::read("libmysofa-sys/libmysofa/tests/tester.sofa").unwrap();
        let sofa = Sofar::open_data(&data).unwrap();
        let filt_len = sofa.filter_len();

        let mut filter = Filter::new(filt_len);
        sofa.filter(0.0, 1.0, 0.0, &mut filter);
    }

    #[test]
    fn debug_tu_berlin() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        std::env::set_current_dir(&cwd).unwrap();

        let path = "libmysofa-sys/libmysofa/tests/TU-Berlin_QU_KEMAR_anechoic_radius_0.5m.sofa";
        let data = std::fs::read(path).unwrap();

        match hdf::parse(&data) {
            Ok(obj) => {
                assert_eq!(obj.name, "root");
            }
            Err(e) => {
                panic!("Parse failed: {:?}", e);
            }
        }
    }
}
