use cmake::Config;
use system_deps::{BuildInternalClosureError, Library};

use std::env;

pub fn build_from_src(lib: &str, version: &str) -> Result<Library, BuildInternalClosureError> {
    let mut config = Config::new("libmysofa");
    let z_root = env::var_os("DEP_Z_ROOT").unwrap();

    let dst = config
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_STATIC_LIBS", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CODE_COVERAGE", "OFF")
        .define("ADDRESS_SANITIZE", "OFF")
        .define("ZLIB_ROOT", z_root)
        .profile("Release")
        .build();

    let pkg_dir = dst.join("lib/pkgconfig");
    Library::from_internal_pkg_config(pkg_dir, lib, version)
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
