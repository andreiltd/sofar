#![cfg(feature = "dsp")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::mpsc::sync_channel;

use sofar::filter::Filter;
use sofar::render::{FilterTransition, Renderer, RendererPlan};

/// Thread-local allocation and deallocation counts for a measured operation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Allocations {
    /// Allocation requests, including reallocations.
    alloc: usize,
    /// Deallocation requests, including the old-storage side of reallocations.
    dealloc: usize,
}

thread_local! {
    static COUNT: Cell<Option<Allocations>> = const { Cell::new(None) };
}

/// System allocator wrapper that counts operations only inside measured regions.
struct CountingAllocator;

fn count(alloc: usize, dealloc: usize) {
    let _ = COUNT.try_with(|counter| {
        let Some(mut counts) = counter.get() else {
            return;
        };

        counts.alloc += alloc;
        counts.dealloc += dealloc;
        counter.set(Some(counts));
    });
}

// SAFETY: every allocation operation is forwarded unchanged to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(1, 0);

        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(1, 0);

        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(1, 1);

        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        count(0, 1);

        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn without_allocations<T>(operation: impl FnOnce() -> T) -> T {
    COUNT.set(Some(Allocations::default()));

    let result = operation();

    let counts = COUNT.replace(None).unwrap();
    assert_eq!(counts, Allocations::default());

    result
}

/// Build a seven-tap filter whose gains and delays vary with `i`.
fn varying_filter(i: usize) -> Filter {
    let mut filter = Filter::new(7);
    filter.left.fill(i as f32 / 100.0);
    filter.right.fill(-(i as f32) / 100.0);
    filter.ldelay = (i % 17) as f32 / 1024.0;
    filter.rdelay = (i % 9) as f32 / 1024.0;

    filter
}

/// Build the seven-tap layout shared by allocation tests.
fn plan() -> RendererPlan {
    RendererPlan::builder(7)
        .with_sample_rate(1024.0)
        .with_partition_len(4)
        .build()
        .unwrap()
}

#[test]
fn steady_state_set_filter_neither_allocates_nor_frees_storage() {
    for transition in [FilterTransition::Immediate, FilterTransition::crossfade(9)] {
        let mut renderer = Renderer::builder(&plan())
            .with_filter_transition(transition)
            .with_max_delay(16.0 / 1024.0)
            .build()
            .unwrap();

        let filters: Vec<_> = (0..100).map(varying_filter).collect();

        // Four updates without processing fill every slot a crossfade can use.
        for filter in &filters[..4] {
            renderer.set_filter(filter).unwrap();
        }

        let mut left = [0.0; 4];
        let mut right = [0.0; 4];

        for (i, filter) in filters.iter().enumerate() {
            without_allocations(|| {
                renderer.set_filter(filter).unwrap();

                // Vary how many fades complete between updates.
                for _ in 0..i % 4 {
                    renderer
                        .process_block(&[1.0; 4], &mut left, &mut right)
                        .unwrap();
                }
            });
        }
    }
}

#[test]
fn callbacks_neither_allocate_nor_free_storage_on_first_use_or_during_updates() {
    for transition in [FilterTransition::Immediate, FilterTransition::crossfade(9)] {
        let (mut publisher, mut renderer) = Renderer::builder(&plan())
            .with_filter_transition(transition)
            .with_max_delay(16.0 / 1024.0)
            .build()
            .unwrap()
            .into_realtime();

        let (commands, receive) = sync_channel::<bool>(1);
        let (finished, complete) = sync_channel(1);

        // A fresh audio thread catches hidden first-use allocations.
        let audio = std::thread::spawn(move || {
            let mut left = [0.0; 4];
            let mut right = [0.0; 4];

            for add in receive {
                without_allocations(|| {
                    if add {
                        renderer.process_block_add(&[1.0; 4], &mut left, &mut right)
                    } else {
                        renderer.process_block(&[1.0; 4], &mut left, &mut right)
                    }
                })
                .unwrap();

                finished.send(()).unwrap();
            }

            without_allocations(|| renderer.reset());

            // Destroy the endpoint on the test thread, outside the measured callback.
            renderer
        });

        commands.send(false).unwrap();
        complete.recv().unwrap();

        for i in 0..100 {
            publisher.publish_filter(&varying_filter(i)).unwrap();

            commands.send(i % 2 == 0).unwrap();
            complete.recv().unwrap();
        }

        // Let delayed and queued transitions finish without more publications.
        for _ in 0..32 {
            commands.send(false).unwrap();
            complete.recv().unwrap();
        }

        drop(commands);
        let renderer = audio.join().unwrap();

        publisher.reclaim();
        drop(renderer);
        drop(publisher);
    }
}

#[test]
fn first_callback_with_a_published_filter_needs_no_thread_local_initialization() {
    let plan = RendererPlan::builder(1)
        .with_partition_len(4)
        .build()
        .unwrap();

    let (mut publisher, mut renderer) = Renderer::new(&plan).into_realtime();

    let mut filter = Filter::new(1);
    filter.left[0] = 1.0;
    filter.right[0] = 1.0;

    publisher.publish_filter(&filter).unwrap();

    let audio = std::thread::spawn(move || {
        let mut left = [0.0; 4];

        without_allocations(|| renderer.process_block(&[1.0; 4], &mut left, &mut [0.0; 4]))
            .unwrap();
        assert_eq!(left, [1.0; 4]);

        renderer
    });

    drop(audio.join().unwrap());
    publisher.reclaim();
}
