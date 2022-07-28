# Sofar
This crate provides high level bindings to [`libmysofa`](https://github.com/hoene/libmysofa)
API allows to read `HRTF` filters from `SOFA` files (Spatially Oriented Format
for Acoustics).

# Example

```rust
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

// apply_delays();
// apply_filters();
```

# License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.
