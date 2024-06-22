use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, StreamConfig};

use anyhow::{bail, Context, Error};
use hound::WavReader;

use sofar::reader::{Filter, OpenOptions, Sofar};
use sofar::render::Renderer;

use ringbuf::{traits::*, HeapRb};

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

    println!("Wave file spec: {:?}", spec);

    let sofa = OpenOptions::new()
        .sample_rate(spec.sample_rate as f32)
        .open(sofa)
        .context("Open sofa file failed")?;

    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();

    let config = device.default_output_config().unwrap();
    println!("Default output config: {:?}", config);

    let mut stream_config = StreamConfig::from(config.clone());
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
    let sample_rate = config.sample_rate.0 as f32;
    let filt_len = sofa.filter_len();

    let mut filter = Filter::new(filt_len);
    sofa.filter(1.0, 0.0, 0.0, &mut filter);

    let mut left = vec![0.0; BLOCK_LEN];
    let mut right = vec![0.0; BLOCK_LEN];

    let render = Renderer::builder(filt_len)
        .with_sample_rate(sample_rate)
        .with_partition_len(64)
        .build()
        .unwrap();

    let render = Arc::new(Mutex::new(render));
    let render_clone = render.clone();

    let eos = Arc::new((Mutex::new(false), Condvar::new()));
    let eos_clone = Arc::clone(&eos);

    let ringbuf = HeapRb::new(BLOCK_LEN * 4);
    let (mut producer, mut consumer) = ringbuf.split();

    for _ in 0..BLOCK_LEN {
        producer.try_push(0.0).unwrap();
    }

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let left_samples = reader.samples::<f32>().len();
            let data_samples = data.len();

            if left_samples < BLOCK_LEN {
                let (lock, cvar) = &*eos_clone;
                let mut eos = lock.lock().unwrap();

                *eos = true;
                cvar.notify_one();

                return;
            }

            while data_samples >= consumer.occupied_len() {
                let src = reader
                    .samples::<f32>()
                    .take(BLOCK_LEN)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();

                render
                    .lock()
                    .unwrap()
                    .process_block(src, &mut left, &mut right)
                    .unwrap();

                for (l, r) in Iterator::zip(left.iter(), right.iter()) {
                    producer.try_push(*l).unwrap();
                    producer.try_push(*r).unwrap();
                }
            }

            for dst in data.chunks_exact_mut(2) {
                dst[0] = consumer.try_pop().unwrap();
                dst[1] = consumer.try_pop().unwrap();
            }
        },
        |err| eprintln!("An error occurred on stream: {}", err),
        None,
    )?;

    stream.play()?;

    thread::spawn(move || {
        let mut x = 1.0;
        let mut y = 0.0;
        let z = 0.0;

        loop {
            // rotate clockwise: https://en.wikipedia.org/wiki/Rotation_matrix
            x = x * f32::cos(ROTATION) + y * f32::sin(ROTATION);
            y = -x * f32::sin(ROTATION) + y * f32::cos(ROTATION);

            println!("Pos: x: {x}, y: {y}");

            sofa.filter(x, y, z, &mut filter);
            render_clone.lock().unwrap().set_filter(&filter).unwrap();

            thread::sleep(time::Duration::from_millis(50));
        }
    });

    let (lock, cvar) = &*eos;
    let mut eos = lock.lock().unwrap();

    while !(*eos) {
        eos = cvar.wait(eos).unwrap();
    }

    Ok(())
}
