use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sofar::{
    filter::Filter,
    render::{FilterTransition, Renderer, RendererPlan},
};
use std::hint::black_box;

const BENCH_FILTER_LEN: usize = 1024;
const BENCH_PARTITION_LEN: usize = 64;
const BENCH_FRAMES: usize = 1024;

fn bench_renderer(b: &mut Bencher, blocks: usize, block_len: usize, filt_len: usize) {
    let mut filt = Filter::new(filt_len);

    fill_signal(&mut filt.left);
    fill_signal(&mut filt.right);

    let mut input = vec![0.0; blocks * block_len];
    let mut left = vec![0.0; blocks * block_len];
    let mut right = vec![0.0; blocks * block_len];

    fill_signal(&mut input);

    let plan = RendererPlan::builder(filt_len)
        .with_partition_len(block_len)
        .build()
        .unwrap();

    let mut renderer = Renderer::new(&plan);

    renderer.set_filter(&filt).unwrap();

    b.iter(|| renderer.process_block(&input, &mut left, &mut right));
}

fn bench_filter_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("Filter Lengths");

    for i in [8, 16, 32, 64, 128, 256, 1024, 4096, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("length", i), i, |b, i| {
            bench_renderer(b, 1, 1024, *i)
        });
    }

    group.finish();
}

fn bench_block_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("Block Lengths");

    for i in [8, 16, 32, 64, 128, 256, 1024, 4096, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("length", i), i, |b, i| {
            bench_renderer(b, 1, *i, 1024)
        });
    }

    group.finish();
}

fn bench_filter_preparation(c: &mut Criterion) {
    let mut filt = Filter::new(BENCH_FILTER_LEN);

    fill_signal(&mut filt.left);
    fill_signal(&mut filt.right);

    let plan = RendererPlan::builder(BENCH_FILTER_LEN)
        .with_partition_len(BENCH_PARTITION_LEN)
        .build()
        .unwrap();

    let mut transform = plan.filter_transform();

    c.bench_function("Prepare Filter", |b| {
        b.iter(|| transform.prepare(&filt).unwrap())
    });
}

fn bench_filter_updates(c: &mut Criterion) {
    let mut filter = Filter::new(BENCH_FILTER_LEN);
    fill_signal(&mut filter.left);
    fill_signal(&mut filter.right);

    let plan = RendererPlan::builder(BENCH_FILTER_LEN)
        .with_partition_len(BENCH_PARTITION_LEN)
        .build()
        .unwrap();

    let mut raw_renderer = Renderer::new(&plan);
    raw_renderer.set_filter(&filter).unwrap();

    let mut transform = plan.filter_transform();
    let prepared_filter = transform.prepare(&filter).unwrap();

    let mut prepared_renderer = Renderer::new(&plan);

    drop(
        prepared_renderer
            .set_prepared_filter(prepared_filter.clone())
            .unwrap(),
    );

    let mut group = c.benchmark_group("Filter Updates");

    group.bench_function(BenchmarkId::new("set_filter", "time-domain"), |b| {
        b.iter(|| raw_renderer.set_filter(black_box(&filter)).unwrap())
    });

    group.bench_function(BenchmarkId::new("set_prepared_filter", "prepared"), |b| {
        b.iter(|| {
            prepared_renderer
                .set_prepared_filter(black_box(prepared_filter.clone()))
                .unwrap()
        })
    });

    group.finish();
}

fn bench_renderer_construction(c: &mut Criterion) {
    let plan = RendererPlan::builder(BENCH_FILTER_LEN)
        .with_partition_len(BENCH_PARTITION_LEN)
        .build()
        .unwrap();

    let crossfade = FilterTransition::crossfade(BENCH_PARTITION_LEN);

    let mut group = c.benchmark_group("Renderer Construction");

    // "direct" includes FFT planning; "plan" measures only per-source construction.
    group.bench_function(BenchmarkId::new("direct", "immediate"), |b| {
        b.iter(|| {
            let plan = RendererPlan::builder(BENCH_FILTER_LEN)
                .with_partition_len(BENCH_PARTITION_LEN)
                .build()
                .unwrap();

            black_box(Renderer::new(&plan))
        })
    });

    group.bench_function(BenchmarkId::new("plan", "immediate"), |b| {
        b.iter(|| black_box(Renderer::new(&plan)))
    });

    group.bench_function(BenchmarkId::new("direct", "crossfade"), |b| {
        b.iter(|| {
            let plan = RendererPlan::builder(BENCH_FILTER_LEN)
                .with_partition_len(BENCH_PARTITION_LEN)
                .build()
                .unwrap();

            black_box(
                Renderer::builder(&plan)
                    .with_filter_transition(crossfade)
                    .build()
                    .unwrap(),
            )
        })
    });

    group.bench_function(BenchmarkId::new("plan", "crossfade"), |b| {
        b.iter(|| {
            black_box(
                Renderer::builder(&plan)
                    .with_filter_transition(crossfade)
                    .build()
                    .unwrap(),
            )
        })
    });

    group.finish();
}

fn bench_processing_modes(c: &mut Criterion) {
    let mut initial_filter = Filter::new(BENCH_FILTER_LEN);
    let mut target_filter = Filter::new(BENCH_FILTER_LEN);
    fill_signal(&mut initial_filter.left);
    fill_signal(&mut initial_filter.right);
    fill_signal(&mut target_filter.left);
    fill_signal(&mut target_filter.right);

    target_filter.left.reverse();
    target_filter.right.reverse();

    let mut input = vec![0.0; BENCH_FRAMES];
    fill_signal(&mut input);

    let plan = RendererPlan::builder(BENCH_FILTER_LEN)
        .with_partition_len(BENCH_PARTITION_LEN)
        .build()
        .unwrap();

    let mut overwrite_renderer = Renderer::new(&plan);
    overwrite_renderer.set_filter(&initial_filter).unwrap();

    let mut overwrite_left = vec![0.0; BENCH_FRAMES];
    let mut overwrite_right = vec![0.0; BENCH_FRAMES];

    let mut additive_renderer = Renderer::new(&plan);
    additive_renderer.set_filter(&initial_filter).unwrap();

    let mut additive_left = vec![0.0; BENCH_FRAMES];
    let mut additive_right = vec![0.0; BENCH_FRAMES];

    // Keep the fade active throughout measurement without timing filter installation.
    let mut crossfade_renderer = Renderer::builder(&plan)
        .with_filter_transition(FilterTransition::crossfade(usize::MAX))
        .build()
        .unwrap();
    crossfade_renderer.set_filter(&initial_filter).unwrap();
    crossfade_renderer.set_filter(&target_filter).unwrap();

    let mut crossfade_left = vec![0.0; BENCH_FRAMES];
    let mut crossfade_right = vec![0.0; BENCH_FRAMES];

    let mut group = c.benchmark_group("Processing Modes");
    group.throughput(Throughput::Elements(BENCH_FRAMES as u64));

    group.bench_function(BenchmarkId::new("process_block", "immediate"), |b| {
        b.iter(|| {
            overwrite_renderer
                .process_block(
                    black_box(&input),
                    black_box(&mut overwrite_left),
                    black_box(&mut overwrite_right),
                )
                .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("process_block_add", "immediate"), |b| {
        b.iter(|| {
            additive_renderer
                .process_block_add(
                    black_box(&input),
                    black_box(&mut additive_left),
                    black_box(&mut additive_right),
                )
                .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("process_block", "crossfade-active"), |b| {
        b.iter(|| {
            crossfade_renderer
                .process_block(
                    black_box(&input),
                    black_box(&mut crossfade_left),
                    black_box(&mut crossfade_right),
                )
                .unwrap()
        })
    });

    group.finish();
}

fn fill_signal(buf: &mut [f32]) {
    for (i, sample) in buf.iter_mut().enumerate() {
        *sample = ((i * 17 + 13) % 101) as f32 / 101.0;
    }
}

criterion_group!(
    benches,
    bench_block_len,
    bench_filter_len,
    bench_filter_preparation,
    bench_filter_updates,
    bench_renderer_construction,
    bench_processing_modes
);
criterion_main!(benches);
