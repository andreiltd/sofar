//! Regression tests for SOFA files written through netCDF4 / HDF5 1.14
//! (e.g. SOFASonix exports from the ASH Toolset).
//!
//! The fixture is a minimal SimpleFreeFieldHRIR set (M=8, R=2, N=16,
//! 48 kHz, unit impulses) written by SOFASonix 1.0.7 via netCDF4 4.9.3 /
//! HDF5 1.14.6. Its object headers use attribute creation order tracking
//! and end with gaps before the chunk checksum, which previously made
//! `SourcePosition` unparsable and `open` fail with `LookupBuildFailed`.

use sofar::reader::{Filter, OpenOptions};

fn fixture() -> String {
    format!(
        "{}/tests/data/sofasonix_netcdf4.sofa",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn open_netcdf4_written_sofa() {
    let sofa = OpenOptions::new()
        .sample_rate(48000.0)
        .open(fixture())
        .expect("open netCDF4-written SOFA");

    assert_eq!(sofa.filter_len(), 16);

    let mut filter = Filter::new(sofa.filter_len());
    sofa.filter(1.0, 0.0, 0.0, &mut filter);
    let energy: f32 = filter.left.iter().map(|x| x * x).sum();
    assert!(energy > 0.0, "front HRIR should not be silent");
}

#[test]
fn parse_unlimited_dimension() {
    // The netCDF4 string dimension `S` is unlimited: its dataspace stores
    // the H5S_UNLIMITED sentinel as maximum size, which must not be
    // rejected by the dimension size sanity check.
    let data = std::fs::read(fixture()).unwrap();
    let parsed = sofar::hdf::parse_with_children(&data).unwrap();
    let s = parsed
        .get_child("S")
        .expect("S dimension present")
        .expect("S dimension parses");
    assert_eq!(s.ds.dimension_size.as_slice(), &[0]);
}
