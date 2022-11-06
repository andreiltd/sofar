use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use sofar::{reader::Filter, render::Renderer};

use rand::Rng;

fn bench_renderer(b: &mut Bencher, blocks: usize, block_len: usize, filt_len: usize) {
    let mut filt = Filter::new(filt_len);

    rand::thread_rng().fill(&mut *filt.left);
    rand::thread_rng().fill(&mut *filt.right);

    let mut input = vec![0.0; blocks * block_len];
    let mut left = vec![0.0; blocks * block_len];
    let mut right = vec![0.0; blocks * block_len];

    rand::thread_rng().fill(input.as_mut_slice());

    let mut renderer = Renderer::builder(filt_len)
        .with_partition_len(block_len)
        .build()
        .unwrap();

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

criterion_group!(benches, bench_block_len, bench_filter_len);
criterion_main!(benches);
