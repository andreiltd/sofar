use cmake::Config;
use std::env;

fn main() {
    if pkg_config::Config::new()
        .atleast_version("1.0")
        .find("mysofa")
        .is_ok()
    {
        println!("cargo:rustc-link-lib=mysofa");
        return;
    }

    let mut config = Config::new("libmysofa");
    let z_root = env::var_os("DEP_Z_ROOT").unwrap();

    let dst = config
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_STATIC_LIBS", "ON")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("CODE_COVERAGE", "OFF")
        .define("ADDRESS_SANITIZE", "OFF")
        .define("ZLIB_ROOT", z_root)
        .profile("Release")
        .build();

    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=libmysofa/src/hrtf/mysofa.h");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mysofa");
    println!("cargo:rustc-link-lib=static=z");
    println!("cargo:outdir={}", out_dir);
}
