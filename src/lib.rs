pub use crate::{error::Error, sofar::Filter, sofar::OpenOptions, sofar::Sofar};

mod error;
mod sofar;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        std::env::set_current_dir(cwd).unwrap();

        let sofa = Sofar::open("libmysofa-sys/libmysofa/tests/tester.sofa").unwrap();
        let filt_len = sofa.filter_len();

        let mut left = vec![0.0; filt_len];
        let mut right = vec![0.0; filt_len];
        let mut delays = [0.0; 2];

        let filter = Filter {
            left: left.as_mut_slice(),
            right: right.as_mut_slice(),
            delays: &mut delays,
        };

        sofa.filter(0.0, 1.0, 0.0, filter);
    }
}
