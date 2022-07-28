#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_int, c_short, c_uint, c_ulong, c_void};

pub const MYSOFA_DEFAULT_NEIGH_STEP_ANGLE: f64 = 0.5;
pub const MYSOFA_DEFAULT_NEIGH_STEP_RADIUS: f64 = 0.01;

pub type size_t = c_ulong;
pub type wchar_t = c_int;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_ATTRIBUTE {
    pub next: *mut MYSOFA_ATTRIBUTE,
    pub name: *mut c_char,
    pub value: *mut c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_ARRAY {
    pub values: *mut f32,
    pub elements: c_uint,
    pub attributes: *mut MYSOFA_ATTRIBUTE,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_VARIABLE {
    pub next: *mut MYSOFA_VARIABLE,
    pub name: *mut c_char,
    pub value: *mut MYSOFA_ARRAY,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_HRTF {
    pub I: c_uint,
    pub C: c_uint,
    pub R: c_uint,
    pub E: c_uint,
    pub N: c_uint,
    pub M: c_uint,

    pub ListenerPosition: MYSOFA_ARRAY,
    pub ReceiverPosition: MYSOFA_ARRAY,
    pub SourcePosition: MYSOFA_ARRAY,
    pub EmitterPosition: MYSOFA_ARRAY,
    pub ListenerUp: MYSOFA_ARRAY,
    pub ListenerView: MYSOFA_ARRAY,
    pub DataIR: MYSOFA_ARRAY,
    pub DataSamplingRate: MYSOFA_ARRAY,
    pub DataDelay: MYSOFA_ARRAY,
    pub attributes: *mut MYSOFA_ATTRIBUTE,
    pub variables: *mut MYSOFA_VARIABLE,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_LOOKUP {
    pub kdtree: *mut c_void,
    pub radius_min: f32,
    pub radius_max: f32,
    pub theta_min: f32,
    pub theta_max: f32,
    pub phi_min: f32,
    pub phi_max: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_NEIGHBORHOOD {
    pub elements: c_int,
    pub index: *mut c_int,
}

pub const MYSOFA_OK: c_int = 0;
pub const MYSOFA_INTERNAL_ERROR: c_int = -1;
pub const MYSOFA_INVALID_FORMAT: c_int = 10000;
pub const MYSOFA_UNSUPPORTED_FORMAT: c_int = 10001;
pub const MYSOFA_NO_MEMORY: c_int = 10002;
pub const MYSOFA_READ_ERROR: c_int = 10003;
pub const MYSOFA_INVALID_ATTRIBUTES: c_int = 10004;
pub const MYSOFA_INVALID_DIMENSIONS: c_int = 10005;
pub const MYSOFA_INVALID_DIMENSION_LIST: c_int = 10006;
pub const MYSOFA_INVALID_COORDINATE_TYPE: c_int = 10007;
pub const MYSOFA_ONLY_EMITTER_WITH_ECI_SUPPORTED: c_int = 10008;
pub const MYSOFA_ONLY_DELAYS_WITH_IR_OR_MR_SUPPORTED: c_int = 10009;
pub const MYSOFA_ONLY_THE_SAME_SAMPLING_RATE_SUPPORTED: c_int = 10010;
pub const MYSOFA_RECEIVERS_WITH_RCI_SUPPORTED: c_int = 10011;
pub const MYSOFA_RECEIVERS_WITH_CARTESIAN_SUPPORTED: c_int = 10012;
pub const MYSOFA_INVALID_RECEIVER_POSITIONS: c_int = 10013;
pub const MYSOFA_ONLY_SOURCES_WITH_MC_SUPPORTED: c_int = 10014;

extern "C" {
    pub fn mysofa_load(filename: *const c_char, err: *mut c_int) -> *mut MYSOFA_HRTF;
    pub fn mysofa_check(hrtf: *mut MYSOFA_HRTF) -> c_int;
    pub fn mysofa_getAttribute(attr: *mut MYSOFA_ATTRIBUTE, name: *mut c_char) -> *mut c_char;
    pub fn mysofa_tospherical(hrtf: *mut MYSOFA_HRTF);
    pub fn mysofa_tocartesian(hrtf: *mut MYSOFA_HRTF);
    pub fn mysofa_free(hrtf: *mut MYSOFA_HRTF);
    pub fn mysofa_lookup_init(hrtf: *mut MYSOFA_HRTF) -> *mut MYSOFA_LOOKUP;
    pub fn mysofa_lookup(lookup: *mut MYSOFA_LOOKUP, coordinate: *mut f32) -> c_int;
    pub fn mysofa_lookup_free(lookup: *mut MYSOFA_LOOKUP);
    pub fn mysofa_neighborhood_init(
        hrtf: *mut MYSOFA_HRTF,
        lookup: *mut MYSOFA_LOOKUP,
    ) -> *mut MYSOFA_NEIGHBORHOOD;
    pub fn mysofa_neighborhood_init_withstepdefine(
        hrtf: *mut MYSOFA_HRTF,
        lookup: *mut MYSOFA_LOOKUP,
        neighbor_angle_step: f32,
        neighbor_radius_step: f32,
    ) -> *mut MYSOFA_NEIGHBORHOOD;
    pub fn mysofa_neighborhood(neighborhood: *mut MYSOFA_NEIGHBORHOOD, pos: c_int) -> *mut c_int;
    pub fn mysofa_neighborhood_free(neighborhood: *mut MYSOFA_NEIGHBORHOOD);
    pub fn mysofa_interpolate(
        hrtf: *mut MYSOFA_HRTF,
        cordinate: *mut f32,
        nearest: c_int,
        neighborhood: *mut c_int,
        fir: *mut f32,
        delays: *mut f32,
    ) -> *mut f32;
    pub fn mysofa_resample(hrtf: *mut MYSOFA_HRTF, samplerate: f32) -> c_int;
    pub fn mysofa_loudness(hrtf: *mut MYSOFA_HRTF) -> f32;
    pub fn mysofa_minphase(hrtf: *mut MYSOFA_HRTF, threshold: f32) -> c_int;
    pub fn mysofa_cache_lookup(filename: *const c_char, samplerate: f32) -> *mut MYSOFA_EASY;
    pub fn mysofa_cache_store(
        arg1: *mut MYSOFA_EASY,
        filename: *const c_char,
        samplerate: f32,
    ) -> *mut MYSOFA_EASY;
    pub fn mysofa_cache_release(arg1: *mut MYSOFA_EASY);
    pub fn mysofa_cache_release_all();
    pub fn mysofa_c2s(values: *mut f32);
    pub fn mysofa_s2c(values: *mut f32);
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MYSOFA_EASY {
    pub hrtf: *mut MYSOFA_HRTF,
    pub lookup: *mut MYSOFA_LOOKUP,
    pub neighborhood: *mut MYSOFA_NEIGHBORHOOD,
    pub fir: *mut f32,
}

extern "C" {
    pub fn mysofa_open(
        filename: *const c_char,
        samplerate: f32,
        filterlength: *mut c_int,
        err: *mut c_int,
    ) -> *mut MYSOFA_EASY;

    pub fn mysofa_open_no_norm(
        filename: *const c_char,
        samplerate: f32,
        filterlength: *mut c_int,
        err: *mut c_int,
    ) -> *mut MYSOFA_EASY;

    pub fn mysofa_open_advanced(
        filename: *const c_char,
        samplerate: f32,
        filterlength: *mut c_int,
        err: *mut c_int,
        norm: bool,
        neighbor_angle_step: f32,
        neighbor_radius_step: f32,
    ) -> *mut MYSOFA_EASY;

    pub fn mysofa_open_cached(
        filename: *const c_char,
        samplerate: f32,
        filterlength: *mut c_int,
        err: *mut c_int,
    ) -> *mut MYSOFA_EASY;

    pub fn mysofa_getfilter_short(
        easy: *mut MYSOFA_EASY,
        x: f32,
        y: f32,
        z: f32,
        IRleft: *mut c_short,
        IRright: *mut c_short,
        delayLeft: *mut c_int,
        delayRight: *mut c_int,
    );

    pub fn mysofa_getfilter_float(
        easy: *mut MYSOFA_EASY,
        x: f32,
        y: f32,
        z: f32,
        IRleft: *mut f32,
        IRright: *mut f32,
        delayLeft: *mut f32,
        delayRight: *mut f32,
    );

    pub fn mysofa_getfilter_float_nointerp(
        easy: *mut MYSOFA_EASY,
        x: f32,
        y: f32,
        z: f32,
        IRleft: *mut f32,
        IRright: *mut f32,
        delayLeft: *mut f32,
        delayRight: *mut f32,
    );

    pub fn mysofa_close(easy: *mut MYSOFA_EASY);
    pub fn mysofa_close_cached(easy: *mut MYSOFA_EASY);
    pub fn mysofa_getversion(major: *mut c_int, minor: *mut c_int, patch: *mut c_int);
}
