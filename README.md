# Sofar
This crate provides high level bindings to [`libmysofa`](https://github.com/hoene/libmysofa)
API allows to read `HRTF` filters from `SOFA` files (Spatially Oriented Format
for Acoustics).

# Example

```rust
use sofar::{Sofar, Filter};

let sofa = Sofar::open("my/sofa/file.sofa").unwrap();
let filt_len = sofa.filter_len();

let mut filter = Filter::new(filt_len);
sofa.filter(0.0, 1.0, 0.0, &mut filter);

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
