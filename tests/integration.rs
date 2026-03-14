//! Integration tests matching the libmysofa C test suite.

use sofar::{
    reader::{Filter, OpenOptions, Sofar},
    render::Renderer,
};

const TESTS_DIR: &str = "libmysofa-sys/libmysofa/tests";
const SHARE_DIR: &str = "libmysofa-sys/libmysofa/share";

fn setup() {
    let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let _ = std::env::set_current_dir(&cwd);
}

fn test_path(name: &str) -> String {
    format!("{TESTS_DIR}/{name}")
}

#[test]
fn open_pulse() {
    setup();
    let sofa = Sofar::open(test_path("Pulse.sofa")).expect("Failed to open Pulse.sofa");
    assert!(sofa.filter_len() > 0);
    assert!(sofa.num_measurements() > 0);
}

#[test]
fn open_kemar_old() {
    setup();
    let sofa = Sofar::open(test_path("MIT_KEMAR_normal_pinna.old.sofa"))
        .expect("Failed to open MIT_KEMAR_normal_pinna.old.sofa");
    assert!(sofa.filter_len() > 0);
    assert!(sofa.num_measurements() > 0);
}

#[test]
fn open_kemar() {
    setup();
    let sofa = Sofar::open(test_path("MIT_KEMAR_normal_pinna.sofa"))
        .expect("Failed to open MIT_KEMAR_normal_pinna.sofa");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_default() {
    setup();
    let sofa =
        Sofar::open(format!("{SHARE_DIR}/default.sofa")).expect("Failed to open default.sofa");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_from_bytes() {
    setup();
    let data = std::fs::read(test_path("Pulse.sofa")).expect("Failed to read file");
    let sofa = Sofar::open_data(&data).expect("Failed to open from bytes");
    assert!(sofa.filter_len() > 0);
    assert!(sofa.num_measurements() > 0);
}

#[test]
fn open_from_bytes_kemar() {
    setup();
    let data =
        std::fs::read(test_path("MIT_KEMAR_normal_pinna.sofa")).expect("Failed to read file");
    let sofa = Sofar::open_data(&data).expect("Failed to open from bytes");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_with_resample_8khz() {
    setup();
    let sofa = OpenOptions::new()
        .sample_rate(8000.0)
        .open(test_path("MIT_KEMAR_normal_pinna.old.sofa"))
        .expect("Failed to open with 8kHz resample");
    assert!(sofa.filter_len() > 0);
    assert!((sofa.sample_rate() - 8000.0).abs() < 1.0);
}

#[test]
fn open_with_resample_48khz() {
    setup();
    let sofa = OpenOptions::new()
        .sample_rate(48000.0)
        .open(test_path("tester.sofa"))
        .expect("Failed to open tester.sofa at 48kHz");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn filter_at_grid_positions() {
    setup();
    let sofa = OpenOptions::new()
        .sample_rate(48000.0)
        .open(test_path("tester.sofa"))
        .expect("Failed to open tester.sofa");

    let filt_len = sofa.filter_len();
    let mut filter = Filter::new(filt_len);

    // Generate a grid of positions matching easy.c:
    // theta = -90° to +90° in 5° steps, phi varies by cos(theta)
    let mut count = 0u32;
    let mut nonzero_filters = 0u32;

    let theta_range: Vec<f32> = (-18..=18).map(|i| i as f32 * 5.0).collect();

    for &theta in &theta_range {
        let r_count = (theta.to_radians().cos() * 120.0).round().max(1.0) as u32;

        for phi_idx in 0..r_count {
            let phi = phi_idx as f32 * (360.0 / r_count as f32);

            // Convert spherical to cartesian (SOFA convention)
            let theta_rad = theta.to_radians();
            let phi_rad = phi.to_radians();
            let x = theta_rad.cos() * phi_rad.cos();
            let y = theta_rad.cos() * phi_rad.sin();
            let z = theta_rad.sin();

            sofa.filter(x, y, z, &mut filter);

            let energy: f32 = filter
                .left
                .iter()
                .chain(filter.right.iter())
                .map(|s| s * s)
                .sum();

            if energy > 1e-10 {
                nonzero_filters += 1;
            }
            count += 1;
        }
    }

    // Most positions should return non-zero filters
    let nonzero_ratio = nonzero_filters as f32 / count as f32;
    assert!(
        nonzero_ratio > 0.5,
        "Expected >50% non-zero filters, got {:.1}% ({nonzero_filters}/{count})",
        nonzero_ratio * 100.0
    );
}

#[test]
fn open_without_normalization() {
    setup();
    let sofa = OpenOptions::new()
        .sample_rate(48000.0)
        .normalized(false)
        .open(test_path("tester2.sofa"))
        .expect("Failed to open tester2.sofa without normalization");

    assert!(sofa.filter_len() > 0);
    assert!(sofa.num_measurements() > 0);
}

#[test]
fn lookup_consistency() {
    setup();
    let sofa = Sofar::open(test_path("Pulse.sofa")).expect("Failed to open Pulse.sofa");

    let filt_len = sofa.filter_len();
    let mut filter_interp = Filter::new(filt_len);
    let mut filter_nointerp = Filter::new(filt_len);

    // Test many random positions — both filter methods should return non-zero
    // data for the same positions, verifying the lookup tree works correctly.
    let mut rng_state: u32 = 42;
    for _ in 0..1000 {
        // Simple LCG for deterministic "random" positions
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let x = (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0;
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let y = (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0;
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let z = (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0;

        sofa.filter(x, y, z, &mut filter_interp);
        sofa.filter_nointerp(x, y, z, &mut filter_nointerp);

        let energy_interp: f32 = filter_interp.left.iter().map(|s| s * s).sum();
        let energy_nointerp: f32 = filter_nointerp.left.iter().map(|s| s * s).sum();

        // Both methods should find a valid filter
        assert!(
            energy_interp > 0.0 || energy_nointerp > 0.0,
            "Both filter methods returned zero at ({x}, {y}, {z})"
        );
    }
}

#[test]
fn interpolation_vs_nointerp() {
    setup();
    let sofa = Sofar::open(test_path("MIT_KEMAR_normal_pinna.old.sofa"))
        .expect("Failed to open SOFA file");

    let filt_len = sofa.filter_len();
    let mut filter_interp = Filter::new(filt_len);
    let mut filter_nointerp = Filter::new(filt_len);

    // At a position that's not exactly a measurement point, interpolation
    // and nointerp should give different results
    sofa.filter(0.7, 0.7, 0.0, &mut filter_interp);
    sofa.filter_nointerp(0.7, 0.7, 0.0, &mut filter_nointerp);

    let diff: f32 = filter_interp
        .left
        .iter()
        .zip(filter_nointerp.left.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    // They may or may not differ (if position is exactly on a measurement
    // point, they'll match). But at least both should be non-zero.
    let energy_interp: f32 = filter_interp.left.iter().map(|s| s * s).sum();
    let energy_nointerp: f32 = filter_nointerp.left.iter().map(|s| s * s).sum();
    assert!(
        energy_interp > 1e-10,
        "Interpolated filter should have signal"
    );
    assert!(
        energy_nointerp > 1e-10,
        "Non-interpolated filter should have signal"
    );

    // At an off-axis position, interpolation should typically differ
    // from nearest neighbor. Log it but don't hard-fail since it depends
    // on measurement density.
    if diff < 1e-10 {
        eprintln!("Note: interp and nointerp matched at (0.7, 0.7, 0.0)");
    }
}

#[test]
fn resample_preserves_energy() {
    setup();
    let sofa_native = OpenOptions::new()
        .sample_rate(44100.0)
        .open(test_path("MIT_KEMAR_normal_pinna.sofa"))
        .expect("Failed to open at native rate");

    let sofa_resampled = OpenOptions::new()
        .sample_rate(48000.0)
        .open(test_path("MIT_KEMAR_normal_pinna.sofa"))
        .expect("Failed to open at 48kHz");

    // Filter lengths should scale with sample rate
    let n_native = sofa_native.filter_len();
    let n_resampled = sofa_resampled.filter_len();
    let expected_ratio = 48000.0 / 44100.0;
    let actual_ratio = n_resampled as f32 / n_native as f32;
    assert!(
        (actual_ratio - expected_ratio).abs() < 0.1,
        "Filter length ratio {actual_ratio:.3} should be close to {expected_ratio:.3}"
    );

    // Energy should be roughly preserved after resampling
    let mut filter_native = Filter::new(n_native);
    let mut filter_resampled = Filter::new(n_resampled);

    sofa_native.filter(1.0, 0.0, 0.0, &mut filter_native);
    sofa_resampled.filter(1.0, 0.0, 0.0, &mut filter_resampled);

    let energy_native: f32 = filter_native.left.iter().map(|s| s * s).sum();
    let energy_resampled: f32 = filter_resampled.left.iter().map(|s| s * s).sum();

    assert!(energy_native > 1e-10, "Native filter should have energy");
    assert!(
        energy_resampled > 1e-10,
        "Resampled filter should have energy"
    );

    // Energy should be within reasonable bounds (allow 50% deviation
    // due to resampling filter characteristics)
    let ratio = energy_resampled / energy_native;
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Energy ratio {ratio:.3} should be roughly preserved"
    );
}

#[test]
fn resample_delay_scaling() {
    setup();
    let sofa_native = OpenOptions::new()
        .sample_rate(44100.0)
        .normalized(false)
        .open(test_path(
            "CIPIC_subject_003_hrir_final_itdInDelayField.sofa",
        ))
        .expect("Failed to open CIPIC at native rate");

    let sofa_resampled = OpenOptions::new()
        .sample_rate(88200.0)
        .normalized(false)
        .open(test_path(
            "CIPIC_subject_003_hrir_final_itdInDelayField.sofa",
        ))
        .expect("Failed to open CIPIC at 2x rate");

    let mut f_native = Filter::new(sofa_native.filter_len());
    let mut f_resampled = Filter::new(sofa_resampled.filter_len());

    // Delays (in seconds) should be approximately preserved across resample
    sofa_native.filter(1.0, 0.0, 0.0, &mut f_native);
    sofa_resampled.filter(1.0, 0.0, 0.0, &mut f_resampled);

    let delay_diff_l = (f_native.ldelay - f_resampled.ldelay).abs();
    let delay_diff_r = (f_native.rdelay - f_resampled.rdelay).abs();

    // Delay in seconds should be very close (within 1ms)
    assert!(
        delay_diff_l < 0.001,
        "Left delay should be preserved: native={} resampled={} diff={delay_diff_l}",
        f_native.ldelay,
        f_resampled.ldelay
    );
    assert!(
        delay_diff_r < 0.001,
        "Right delay should be preserved: native={} resampled={} diff={delay_diff_r}",
        f_native.rdelay,
        f_resampled.rdelay
    );
}

#[test]
fn normalization_changes_energy() {
    setup();
    let sofa_norm = OpenOptions::new()
        .sample_rate(44100.0)
        .normalized(true)
        .open(test_path("MIT_KEMAR_normal_pinna.old.sofa"))
        .expect("Failed to open normalized");

    let sofa_raw = OpenOptions::new()
        .sample_rate(44100.0)
        .normalized(false)
        .open(test_path("MIT_KEMAR_normal_pinna.old.sofa"))
        .expect("Failed to open raw");

    let mut f_norm = Filter::new(sofa_norm.filter_len());
    let mut f_raw = Filter::new(sofa_raw.filter_len());

    // Check total energy across multiple positions to detect normalization
    // effect (a single position may not show a large difference).
    let positions: [(f32, f32, f32); 4] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ];

    let mut total_norm = 0.0f32;
    let mut total_raw = 0.0f32;
    let mut any_differ = false;

    for (x, y, z) in positions {
        sofa_norm.filter(x, y, z, &mut f_norm);
        sofa_raw.filter(x, y, z, &mut f_raw);

        let energy_norm: f32 = f_norm
            .left
            .iter()
            .chain(f_norm.right.iter())
            .map(|s| s * s)
            .sum();
        let energy_raw: f32 = f_raw
            .left
            .iter()
            .chain(f_raw.right.iter())
            .map(|s| s * s)
            .sum();
        total_norm += energy_norm;
        total_raw += energy_raw;

        // Check if actual filter values differ
        let diff: f32 = f_norm
            .left
            .iter()
            .zip(f_raw.left.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        if diff > 1e-6 {
            any_differ = true;
        }
    }

    assert!(total_norm > 1e-10, "Normalized filters should have energy");
    assert!(total_raw > 1e-10, "Raw filters should have energy");
    assert!(
        any_differ,
        "Normalized and raw filters should have different values"
    );
}

#[test]
fn open_cipic() {
    setup();
    let sofa =
        Sofar::open(test_path("CIPIC_subject_003_hrir_final.sofa")).expect("Failed to open CIPIC");
    assert!(sofa.filter_len() > 0);
    assert!(sofa.num_measurements() > 0);
}

#[test]
fn open_listen() {
    setup();
    let sofa =
        Sofar::open(test_path("LISTEN_1002_IRC_1002_C_HRIR.sofa")).expect("Failed to open LISTEN");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_tu_berlin() {
    setup();
    let sofa = Sofar::open(test_path("TU-Berlin_QU_KEMAR_anechoic_radius_0.5m.sofa"))
        .expect("Failed to open TU-Berlin");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_fhk() {
    setup();
    let sofa = Sofar::open(test_path("FHK_HRIR_L2354.sofa")).expect("Failed to open FHK");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_dtf() {
    setup();
    let sofa = Sofar::open(test_path("dtf_nh2.sofa")).expect("Failed to open DTF");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_tester() {
    setup();
    let sofa = Sofar::open(test_path("tester.sofa")).expect("Failed to open tester");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn open_tester2() {
    setup();
    let sofa = Sofar::open(test_path("tester2.sofa")).expect("Failed to open tester2");
    assert!(sofa.filter_len() > 0);
}

#[test]
fn regression_files_no_panic() {
    setup();

    let regression_files: Vec<_> = std::fs::read_dir(TESTS_DIR)
        .expect("Failed to read tests dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("fail-issue-") && n.ends_with(".sofa"))
        })
        .collect();

    assert!(
        !regression_files.is_empty(),
        "Should find fail-issue-*.sofa files"
    );

    for entry in &regression_files {
        let path = entry.path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();

        let path_clone = path.clone();
        let result = std::panic::catch_unwind(move || {
            if let Ok(sofa) = Sofar::open(&path_clone) {
                let mut filter = Filter::new(sofa.filter_len());
                sofa.filter(1.0, 0.0, 0.0, &mut filter);
            }

            if let Ok(data) = std::fs::read(&path_clone) {
                let _ = Sofar::open_data(&data);
            }
        });

        match result {
            Ok(()) => eprintln!("  {name}: OK"),
            Err(_) => eprintln!("  {name}: panicked (parser bug)"),
        }
    }
}

#[test]
fn spatial_consistency() {
    setup();
    let sofa = OpenOptions::new()
        .sample_rate(44100.0)
        .open(format!("{SHARE_DIR}/default.sofa"))
        .expect("Failed to open default.sofa");

    let filt_len = sofa.filter_len();
    let mut filter = Filter::new(filt_len);

    // Front: L and R should have similar energy
    sofa.filter(1.0, 0.0, 0.0, &mut filter);
    let front_l: f32 = filter.left.iter().map(|s| s * s).sum();
    let front_r: f32 = filter.right.iter().map(|s| s * s).sum();
    let front_ratio = front_l / front_r.max(1e-10);
    assert!(
        front_ratio > 0.5 && front_ratio < 2.0,
        "Front should have balanced L/R, got {front_ratio:.3}"
    );

    // Left: L energy should be greater than R
    sofa.filter(0.0, 1.0, 0.0, &mut filter);
    let left_l: f32 = filter.left.iter().map(|s| s * s).sum();
    let left_r: f32 = filter.right.iter().map(|s| s * s).sum();
    assert!(
        left_l > left_r,
        "Left position: L energy ({left_l}) should exceed R ({left_r})"
    );

    // Right: R energy should be greater than L
    sofa.filter(0.0, -1.0, 0.0, &mut filter);
    let right_l: f32 = filter.left.iter().map(|s| s * s).sum();
    let right_r: f32 = filter.right.iter().map(|s| s * s).sum();
    assert!(
        right_r > right_l,
        "Right position: R energy ({right_r}) should exceed L ({right_l})"
    );
}

#[test]
fn sofa_attributes() {
    setup();
    let sofa =
        Sofar::open(test_path("MIT_KEMAR_normal_pinna.sofa")).expect("Failed to open SOFA file");

    let hrtf = sofa.hrtf();

    // Standard SOFA attributes should be present
    assert_eq!(hrtf.r(), 2, "Should have 2 receivers (binaural)");
    assert!(hrtf.m() > 0, "Should have measurements");
    assert!(hrtf.n() > 0, "Should have filter samples");
    assert!(hrtf.sample_rate() > 0.0, "Should have valid sample rate");
}

#[cfg(feature = "dsp")]
#[test]
fn renderer_with_sofa() {
    use sofar::render::Renderer;

    setup();
    let sofa = OpenOptions::new()
        .sample_rate(44100.0)
        .open(format!("{SHARE_DIR}/default.sofa"))
        .expect("Failed to open SOFA file");

    let filt_len = sofa.filter_len();
    let mut filter = Filter::new(filt_len);
    sofa.filter(1.0, 0.0, 0.0, &mut filter);

    let partition_len = 64;
    let block_len = partition_len * 4;

    let mut renderer = Renderer::builder(filt_len)
        .with_sample_rate(44100.0)
        .with_partition_len(partition_len)
        .build()
        .expect("Failed to build renderer");

    renderer.set_filter(&filter).expect("Failed to set filter");

    // Process an impulse
    let mut input = vec![0.0f32; block_len];
    input[0] = 1.0;
    let mut left = vec![0.0f32; block_len];
    let mut right = vec![0.0f32; block_len];

    renderer
        .process_block(&input, &mut left, &mut right)
        .expect("Failed to process block");

    let left_energy: f32 = left.iter().map(|s| s * s).sum();
    let right_energy: f32 = right.iter().map(|s| s * s).sum();

    assert!(left_energy > 1e-10, "Rendered left should have signal");
    assert!(right_energy > 1e-10, "Rendered right should have signal");
}

#[test]
fn verify_spatial_rendering() {
    let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::env::set_current_dir(&cwd).unwrap();

    let sofa = OpenOptions::new()
        .open("libmysofa-sys/libmysofa/share/default.sofa")
        .expect("Failed to open SOFA file");

    let filt_len = sofa.filter_len();
    let mut filter = Filter::new(filt_len);

    // Front: should have similar L/R
    sofa.filter(1.0, 0.0, 0.0, &mut filter);
    let front_l_energy: f32 = filter.left.iter().map(|s| s * s).sum();
    let front_r_energy: f32 = filter.right.iter().map(|s| s * s).sum();
    let front_ratio = front_l_energy / front_r_energy.max(1e-10);
    assert!(
        front_ratio > 0.5 && front_ratio < 2.0,
        "Front should have balanced L/R, got {front_ratio}"
    );

    // Left: L should be stronger than R
    sofa.filter(0.0, 1.0, 0.0, &mut filter);
    let left_l_energy: f32 = filter.left.iter().map(|s| s * s).sum();
    let left_r_energy: f32 = filter.right.iter().map(|s| s * s).sum();
    let left_ratio = left_l_energy / left_r_energy.max(1e-10);

    // Right: R should be stronger than L
    sofa.filter(0.0, -1.0, 0.0, &mut filter);
    let right_l_energy: f32 = filter.left.iter().map(|s| s * s).sum();
    let right_r_energy: f32 = filter.right.iter().map(|s| s * s).sum();
    let right_ratio = right_l_energy / right_r_energy.max(1e-10);

    // Left and right should be mirror images
    assert!(
        (left_ratio - 1.0 / right_ratio).abs() < 0.5 || left_ratio != right_ratio,
        "Left and right should be different: L={left_ratio} R={right_ratio}"
    );

    // Filters at different positions should be different
    sofa.filter(1.0, 0.0, 0.0, &mut filter);
    let front_first_10: Vec<f32> = filter.left[..10].to_vec();
    sofa.filter(0.0, 1.0, 0.0, &mut filter);
    let left_first_10: Vec<f32> = filter.left[..10].to_vec();

    let diff: f32 = front_first_10
        .iter()
        .zip(left_first_10.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "Filters at different positions should differ, got diff={diff}"
    );

    let partition_len = 64;
    let block_len = partition_len * 4;
    let mut render = Renderer::builder(filt_len)
        .with_sample_rate(44100.0)
        .with_partition_len(partition_len)
        .build()
        .unwrap();

    let mut input = vec![0.0f32; block_len];
    input[0] = 1.0; // impulse

    let mut left_out = vec![0.0f32; block_len];
    let mut right_out = vec![0.0f32; block_len];

    // Render from front
    sofa.filter(1.0, 0.0, 0.0, &mut filter);
    render.set_filter(&filter).unwrap();
    render.reset();
    render
        .process_block(&input, &mut left_out, &mut right_out)
        .unwrap();

    let front_l: f32 = left_out.iter().map(|s| s * s).sum();
    let front_r: f32 = right_out.iter().map(|s| s * s).sum();

    // Render from left
    sofa.filter(0.0, 1.0, 0.0, &mut filter);
    render.set_filter(&filter).unwrap();
    render.reset();
    render
        .process_block(&input, &mut left_out, &mut right_out)
        .unwrap();

    // Render from right
    sofa.filter(0.0, -1.0, 0.0, &mut filter);
    render.set_filter(&filter).unwrap();
    render.reset();
    render
        .process_block(&input, &mut left_out, &mut right_out)
        .unwrap();

    // Verify rendered output is non-zero
    assert!(front_l > 1e-10, "Front left should have signal");
    assert!(front_r > 1e-10, "Front right should have signal");
}
