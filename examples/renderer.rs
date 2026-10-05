use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, StreamConfig};

use anyhow::{Context, Error, bail};
use hound::WavReader;

use sofar::reader::{Filter, OpenOptions, Sofar};
use sofar::render::{Error as RenderError, FilterTransition, Renderer, RendererPlan};

use ringbuf::{HeapRb, traits::*};

use std::sync::{Arc, Condvar, Mutex};
use std::{env, io::Read};
use std::{thread, time};

// Rotation in radians to apply to object position every 50 ms
const ROTATION: f32 = 2.0 / 180.0 * std::f32::consts::PI;
// Single block size in frames
const BLOCK_LEN: usize = 1024;

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        bail!("Usage: {} MONO_WAV_FILE SOFA_FILE", args[0].clone());
    }

    let wav = &args[1];
    let sofa = &args[2];

    let reader = WavReader::open(wav).context("Open wav file failed")?;
    let spec = reader.spec();

    if spec.sample_format != hound::SampleFormat::Float || spec.channels != 1 {
        bail!("Unsupported format, must be F32, mono channel");
    }

    println!("Wave file spec: {spec:?}");

    let sofa = OpenOptions::new()
        .sample_rate(spec.sample_rate as f32)
        .open(sofa)
        .context("Open sofa file failed")?;

    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();

    let config = device.default_output_config().unwrap();
    println!("Default output config: {config:?}");

    let mut stream_config = StreamConfig::from(config);
    stream_config.channels = 2;
    stream_config.buffer_size = BufferSize::Fixed(BLOCK_LEN as u32);

    match config.sample_format() {
        cpal::SampleFormat::F32 => run(&device, &stream_config, sofa, reader),
        fmt => bail!("Unsupported sample format {:?}", fmt),
    }
}

pub fn run<R>(
    device: &cpal::Device,
    config: &StreamConfig,
    sofa: Sofar,
    mut reader: WavReader<R>,
) -> Result<(), Error>
where
    R: Read + Send + 'static,
{
    let sample_rate = config.sample_rate as f32;
    let filt_len = sofa.filter_len();

    let mut input_buf = vec![0.0f32; BLOCK_LEN];
    let mut left = vec![0.0; BLOCK_LEN];
    let mut right = vec![0.0; BLOCK_LEN];

    let plan = RendererPlan::builder(filt_len)
        .with_sample_rate(sample_rate)
        .with_partition_len(64)
        .build()?;

    // The renderer outputs silence until the worker publishes the first filter.
    let (mut publisher, mut render) = Renderer::builder(&plan)
        .with_filter_transition(FilterTransition::crossfade(32))
        .with_max_delay(sofa.max_delay())
        .build()?
        .into_realtime();

    let eos = Arc::new((Mutex::new(false), Condvar::new()));
    let eos_clone = Arc::clone(&eos);

    let ringbuf = HeapRb::new(BLOCK_LEN * 4);
    let (mut producer, mut consumer) = ringbuf.split();

    for _ in 0..BLOCK_LEN {
        let _ = producer.try_push(0.0);
    }

    let stream = device.build_output_stream(
        *config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let data_samples = data.len();

            while data_samples >= consumer.occupied_len() {
                let mut got = 0;

                for s in reader.samples::<f32>().take(BLOCK_LEN).flatten() {
                    input_buf[got] = s;
                    got += 1;
                }

                if got < BLOCK_LEN {
                    let (lock, cvar) = &*eos_clone;

                    if let Ok(mut eos) = lock.lock() {
                        *eos = true;
                        cvar.notify_one();
                    }

                    return;
                }

                render
                    .process_block(&input_buf, &mut left, &mut right)
                    .expect("renderer buffers are partition-aligned");

                for (l, r) in Iterator::zip(left.iter(), right.iter()) {
                    let _ = producer.try_push(*l);
                    let _ = producer.try_push(*r);
                }
            }

            for dst in data.as_chunks_mut::<2>().0 {
                dst[0] = consumer.try_pop().unwrap_or(0.0);
                dst[1] = consumer.try_pop().unwrap_or(0.0);
            }
        },
        |err| eprintln!("An error occurred on stream: {err}"),
        None,
    )?;

    stream.play()?;

    let worker = thread::spawn(move || {
        let mut x: f32 = 1.0;
        let mut y: f32 = 0.0;
        let z: f32 = 0.0;

        let cos_r = f32::cos(ROTATION);
        let sin_r = f32::sin(ROTATION);

        let mut filter = Filter::new(filt_len);

        loop {
            let new_x = x * cos_r + y * sin_r;
            let new_y = -x * sin_r + y * cos_r;
            x = new_x;
            y = new_y;

            println!("Pos: x: {x}, y: {y}");

            sofa.filter(x, y, z, &mut filter);

            match publisher.publish_filter(&filter) {
                Ok(()) => {}
                // The renderer was dropped with the stream.
                Err(RenderError::RendererDisconnected) => break,
                Err(err) => eprintln!("Failed to publish filter: {err}"),
            }

            thread::sleep(time::Duration::from_millis(50));
        }
    });

    let (lock, cvar) = &*eos;
    let mut eos = lock.lock().unwrap();

    while !(*eos) {
        eos = cvar.wait(eos).unwrap();
    }

    // Release the lock first: the stream's callback takes it until it stops.
    drop(eos);

    // Dropping the stream drops the renderer, which stops the worker.
    drop(stream);
    worker.join().expect("filter worker panicked");

    Ok(())
}
