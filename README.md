<div align="center">

<img src="docs/homer-sofar.png"/>

# Sofar
Pure Rust SOFA Reader and HRTF Renderer

</div>

## Features
A pure Rust implementation for reading `HRTF` filters from `SOFA` files
(Spatially Oriented Format for Acoustics).

The [`render`] module implements uniformly partitioned convolution algorithm
for rendering HRTF filters.

Based on the [`libmysofa`] C library by Christian Hoene / Symonics GmbH.

[`libmysofa`]: https://github.com/hoene/libmysofa
[`render`]: `crate::render`

## Example

```rust
use sofar::reader::{Filter, OpenOptions};
use sofar::render::{Renderer, RendererPlan};

// Open sofa file, resample HRTF data if needed to 44_100
let sofa = OpenOptions::new()
    .sample_rate(44100.0)
    .open("my/sofa/file.sofa")
    .unwrap();

let filt_len = sofa.filter_len();
let mut filter = Filter::new(filt_len);

// Get filter at position
sofa.filter(0.0, 1.0, 0.0, &mut filter);

let plan = RendererPlan::builder(filt_len)
    .with_sample_rate(sofa.sample_rate())
    .with_partition_len(64)
    .build()
    .unwrap();

let mut render = Renderer::new(&plan);
render.set_filter(&filter).unwrap();

let input = vec![0.0; 256];
let mut left = vec![0.0; 256];
let mut right = vec![0.0; 256];

// read_input()

render.process_block(&input, &mut left, &mut right).unwrap();
```

Filter delays, which some SOFA files use for interaural time differences, are
ignored unless enabled with
`Renderer::builder(&plan).with_max_delay(sofa.max_delay())`. For several
sources, create one renderer per source from the same plan and mix them with
`process_block_add`.

You can run `cpal` renderer example like this:

``` shell
cargo run --example renderer -- <FILENAME-MONO.wav> libmysofa-sys/libmysofa/share/default.sofa
```

## Real-time filter updates

`set_filter` prepares filters on the calling thread. To keep that work out of
the audio callback, split a renderer with `into_realtime`: a worker thread
publishes filters, and the audio thread adopts them at partition boundaries
without allocating or freeing memory. Continuing the first example:

```rust
use sofar::render::{FilterTransition, Renderer};

let (mut publisher, mut render) = Renderer::builder(&plan)
    .with_filter_transition(FilterTransition::crossfade(32))
    .with_max_delay(sofa.max_delay())
    .build()
    .unwrap()
    .into_realtime();

// Worker thread, whenever the source moves:
sofa.filter(1.0, 0.0, 0.0, &mut filter);
publisher.publish_filter(&filter).unwrap();

// Audio callback:
render.process_block(&input, &mut left, &mut right).unwrap();
```

- **Latest wins:** publishing replaces a filter the renderer has not adopted
  yet; a running crossfade finishes before the next one starts.
- **Validation:** invalid filters are rejected on the worker and never reach
  the renderer. `publish` sends already prepared filters, such as cached ones.
- **Memory:** displaced filters return to the publisher, which frees them when
  publishing or on `reclaim()`.
- **Shutdown:** once the renderer is dropped, publishing returns
  `Error::RendererDisconnected`. Drop the renderer outside the audio callback;
  the publisher can be dropped at any time.

Custom transports can use `Renderer::set_prepared_filter` and
`take_retired_filter`, freeing the filters they return off the audio thread.

## Upgrading from 0.3

- Sample rate and partition length move to `RendererPlan::builder(filter_len)`;
  create renderers from the plan with `Renderer::new` or `Renderer::builder`.
- `with_left_delay`/`with_right_delay` become `with_max_delay(seconds)`, e.g.
  `with_max_delay(sofa.max_delay())`. Filters with longer delays are rejected.
- `process_block` takes slices; `Error` is non-exhaustive with named fields.
- `set_filter` returns an error instead of panicking on mismatched channels.
  Its first few calls allocate; `into_realtime` keeps that off the audio thread.

## Acknowledgments

This project is a Rust port of [libmysofa](https://github.com/hoene/libmysofa),
a C library for reading SOFA files.

- **libmysofa** Copyright © 2016-2017 Symonics GmbH, Christian Hoene (BSD-3-Clause)
- **KD-tree** Copyright © 2007-2011 John Tsiombikas (BSD-3-Clause)

See the [NOTICE](NOTICE) file for full attribution details.

# License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.
