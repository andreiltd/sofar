use system_deps::{BuildInternalClosureError, Library};

use std::env;

pub fn build_from_src(lib: &str, version: &str) -> Result<Library, BuildInternalClosureError> {
    let lib = lib.strip_prefix("lib").unwrap_or(lib);
    let dst = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let z_root = env::var_os("DEP_Z_ROOT").unwrap();
    let z_inc = std::path::PathBuf::from(&z_root).join("include");
    let z_lib = std::path::PathBuf::from(&z_root).join("lib");

    let mut cfg = cc::Build::new();
    cfg.flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .include(z_inc)
        .include("libmysofa/src/hrtf")
        .include("c")
        .file("libmysofa/src/hrtf/cache.c")
        .file("libmysofa/src/hdf/gunzip.c")
        .file("libmysofa/src/hdf/gcol.c")
        .file("libmysofa/src/hrtf/tools.c")
        .file("libmysofa/src/hdf/superblock.c")
        .file("libmysofa/src/hrtf/loudness.c")
        .file("libmysofa/src/hrtf/lookup.c")
        .file("libmysofa/src/hdf/btree.c")
        .file("libmysofa/src/hrtf/minphase.c")
        .file("libmysofa/src/hrtf/neighbors.c")
        .file("libmysofa/src/hrtf/easy.c")
        .file("libmysofa/src/hrtf/check.c")
        .file("libmysofa/src/hrtf/kdtree.c")
        .file("libmysofa/src/hrtf/spherical.c")
        .file("libmysofa/src/hrtf/reader.c")
        .file("libmysofa/src/hdf/fractalhead.c")
        .file("libmysofa/src/hrtf/interpolate.c")
        .file("libmysofa/src/hrtf/resample.c")
        .file("libmysofa/src/hdf/dataobject.c")
        .file("libmysofa/src/resampler/speex_resampler.c")
        .compile(lib);

    Ok(Library {
        name: lib.to_owned(),
        version: version.to_owned(),
        source: system_deps::Source::EnvVariables,
        link_paths: vec![dst, z_lib.to_owned()],
        libs: vec![
            system_deps::InternalLib {
                name: lib.to_owned(),
                is_static_available: false,
            },
            system_deps::InternalLib {
                name: "z".to_owned(),
                is_static_available: false,
            },
        ],
        frameworks: Default::default(),
        framework_paths: Default::default(),
        include_paths: Default::default(),
        defines: Default::default(),
        ld_args: Default::default(),
        statik: true,
    })
}

fn main() {
    let build_internal_key = "SYSTEM_DEPS_LIBMYSOFA_BUILD_INTERNAL";
    let build_internal_val = env::var_os(build_internal_key).unwrap_or("auto".into());

    env::set_var(build_internal_key, build_internal_val);

    system_deps::Config::new()
        .add_build_internal("libmysofa", build_from_src)
        .probe()
        .unwrap();
}
