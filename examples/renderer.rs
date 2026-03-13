use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, StreamConfig};

use anyhow::{Context, Error, bail};
use arc_swap::ArcSwap;
use hound::WavReader;

use sofar::reader::{Filter, OpenOptions, Sofar};
use sofar::render::Renderer;

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
    let sample_rate = config.sample_rate as f32;
    let filt_len = sofa.filter_len();

    let initial_filter = Filter::new(filt_len);

    let mut input_buf = vec![0.0f32; BLOCK_LEN];
    let mut left = vec![0.0; BLOCK_LEN];
    let mut right = vec![0.0; BLOCK_LEN];

    let mut render = Renderer::builder(filt_len)
        .with_sample_rate(sample_rate)
        .with_partition_len(64)
        .build()
        .unwrap();

    render.set_filter(&initial_filter).unwrap();

    let pending_filter: Arc<ArcSwap<Option<Filter>>> = Arc::new(ArcSwap::from_pointee(None));
    let pending_clone = Arc::clone(&pending_filter);

    let eos = Arc::new((Mutex::new(false), Condvar::new()));
    let eos_clone = Arc::clone(&eos);

    let ringbuf = HeapRb::new(BLOCK_LEN * 4);
    let (mut producer, mut consumer) = ringbuf.split();

    for _ in 0..BLOCK_LEN {
        let _ = producer.try_push(0.0);
    }

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let guard = pending_filter.load();
            if let Some(new_filter) = guard.as_ref() {
                let _ = render.set_filter(new_filter);
                pending_filter.store(Arc::new(None));
            }

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

                let _ = render.process_block(&input_buf, &mut left, &mut right);

                for (l, r) in Iterator::zip(left.iter(), right.iter()) {
                    let _ = producer.try_push(*l);
                    let _ = producer.try_push(*r);
                }
            }

            for dst in data.chunks_exact_mut(2) {
                dst[0] = consumer.try_pop().unwrap_or(0.0);
                dst[1] = consumer.try_pop().unwrap_or(0.0);
            }
        },
        |err| eprintln!("An error occurred on stream: {err}"),
        None,
    )?;

    stream.play()?;

    thread::spawn(move || {
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

            let mut new_filter = Filter::new(filt_len);
            new_filter.left.copy_from_slice(&filter.left);
            new_filter.right.copy_from_slice(&filter.right);
            new_filter.ldelay = filter.ldelay;
            new_filter.rdelay = filter.rdelay;

            pending_clone.store(Arc::new(Some(new_filter)));

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
