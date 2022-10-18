pub use crate::{error::Error, sofar::Filter, sofar::OpenOptions, sofar::Sofar};

mod error;
mod sofar;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        cwd.push("..");

        std::env::set_current_dir(cwd).unwrap();

        let sofa = Sofar::open("libmysofa-sys/libmysofa/tests/tester.sofa").unwrap();
        let filt_len = sofa.filter_len();

        let left = vec![0.0; filt_len];
        let right = vec![0.0; filt_len];

        let mut filter = Filter {
            left: left.into_boxed_slice(),
            right: right.into_boxed_slice(),
            ldelay: 0.0,
            rdelay: 0.0,
        };

        sofa.filter(0.0, 1.0, 0.0, &mut filter);
    }
}
