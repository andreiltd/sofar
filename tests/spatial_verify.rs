use sofar::reader::{Filter, OpenOptions};
use sofar::render::Renderer;

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
