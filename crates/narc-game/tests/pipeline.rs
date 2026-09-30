//! End-to-end: mock capture -> DiskCapture -> import -> bake -> evaluate.
use narc_game::capture::{DataLevel, FrameSource};
use narc_game::mock::{self, MockKind};
use narc_game::narc::NarcConfig;
use narc_game::{pipeline, DiskCapture};

#[test]
fn mock_capture_roundtrip_and_pipeline() -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!("narc-game-e2e-{}", std::process::id()));
    let cap_dir = root.join("capture");
    let out = root.join("out");
    mock::generate(&cap_dir, MockKind::Static, 96, 54, 9)?;

    // PNG is 8-bit, depth is exact.
    let cap = DiskCapture::open(&cap_dir)?;
    assert_eq!(cap.len(), 9);
    assert_eq!(cap.level()?, DataLevel::L2Normals);
    let (color, depth, _, cam) = mock::render(MockKind::Static, 0.5, 96, 54)?;
    assert_eq!(cap.depth(4)?.unwrap(), depth);
    let back = cap.color(4)?;
    assert!(back.data.iter().zip(&color.data).all(|(a, b)| (a - b).abs() <= 0.5 / 255.0 + 1e-6));
    let c = cap.camera(4)?.unwrap();
    assert!(narc_game::camera::length(narc_game::camera::sub(c.position, cam.position)) < 1e-9);

    let imp = pipeline::import(&cap)?;
    assert!(imp.geometry_usable && imp.issues.is_empty() && imp.frames_with_camera == 9);

    let bake = pipeline::bake(&cap, &cap_dir, &out, 4, &NarcConfig::default())?;
    assert_eq!(bake.keyframes, vec![0, 4, 8]);
    assert!(bake.cache_entries > 0);

    let r = pipeline::evaluate(&cap, &out, &out, 1)?;
    assert_eq!(r.unseen_frames, vec![1, 2, 3, 5, 6, 7]);
    assert!(r.unseen_frames.iter().all(|f| !r.keyframes.contains(f)));
    let rep = &r.summaries[0];
    assert_eq!(rep.method, "reprojection");
    assert!(rep.baked_psnr_db.unwrap_or(f64::INFINITY) > 60.0, "{:?}", rep.baked_psnr_db);
    assert!(rep.coverage_mean > 0.9);
    for f in ["metrics.json", "metrics.csv", "bake.json", "narc_cache.bin"] { assert!(out.join(f).exists(), "{f}"); }
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn keyframe_selection_leaves_unseen_frames() {
    assert_eq!(pipeline::keyframe_indices(10, 3), vec![0, 3, 6, 9]);
}

#[test]
fn cache_file_roundtrip() -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!("narc-game-cache-{}", std::process::id()));
    let cap_dir = root.join("capture");
    mock::generate(&cap_dir, MockKind::Static, 64, 36, 3)?;
    let cap = DiskCapture::open(&cap_dir)?;
    pipeline::bake(&cap, &cap_dir, &root, 2, &NarcConfig::default())?;
    let a = narc_game::narc::NarcCache::load(&root.join("narc_cache.bin"))?;
    a.save(&root.join("again.bin"))?;
    assert_eq!(std::fs::read(root.join("narc_cache.bin"))?, std::fs::read(root.join("again.bin"))?);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn import_flags_a_transposed_camera_matrix() -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!("narc-game-transposed-{}", std::process::id()));
    mock::generate(&root, MockKind::Static, 96, 54, 25)?;
    let good = pipeline::import(&DiskCapture::open(&root)?)?;
    assert!(pipeline::validate(&DiskCapture::open(&root)?, 5)?.ready);
    assert!(good.pair_check_coverage.unwrap() > 0.8 && good.pair_check_psnr_db.unwrap() > 25.0, "{good:?}");
    // Row-vector matrix exported without transposition.
    for i in 0..25 {
        let p = root.join(narc_game::capture::frame_dir_name(i)).join("metadata.json");
        let mut meta: narc_game::capture::FrameMeta = serde_json::from_slice(&std::fs::read(&p)?)?;
        let cam = meta.camera.as_mut().unwrap();
        let m = cam.world_to_clip;
        cam.world_to_clip = std::array::from_fn(|r| std::array::from_fn(|c| m[c][r]));
        cam.position = None;
        std::fs::write(&p, serde_json::to_vec(&meta)?)?;
    }
    let bad = pipeline::import(&DiskCapture::open(&root)?);
    match bad {
        Ok(r) => assert!(r.issues.iter().any(|i| i.contains("camera")), "transposed matrix not flagged: {r:?}"),
        Err(_) => {}
    }
    assert!(pipeline::validate(&DiskCapture::open(&root)?, 5).map_or(true, |v| !v.ready), "validate-capture accepted a transposed matrix");
    std::fs::remove_dir_all(root)?;
    Ok(())
}
