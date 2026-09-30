# Limitations

## Dense cache

- **Size grows multiplicatively.** `6 x X x Y x Z x V x L x 16 bytes`. The quality
  configuration already needs 1.5 GiB. This does not scale to real scenes.
- **Hit rate carries no information.** Every query lands in a baked cell; quality
  is PSNR and max error.
- **Restricted angular domain.** View and light follow fixed-elevation orbits, so
  only azimuth is binned. A full sphere (octahedral mapping) multiplies the size.
- **Unused cells.** Surface height stays in [-0.25, 0.25] while the grid covers
  [-1, 1]: about three quarters of the z cells are never queried.
- **FP32 RGBx payload.** 16 bytes per entry, one channel unused.

## Measurement

- The teacher uses random smooth weights. A trained, higher-frequency appearance
  function would be harder to cache.
- FULL is a plain FP32 MLP; see [Benchmark Methodology](Benchmark-Methodology).
- GPU activity tracing is unavailable under WSL2; the zero-copy check uses the
  driver API trace.

## Roadmap

1. Residual network: cached RGB or latent plus current features, 32-32-3 MLP
   trained against the teacher.
2. 16-D latent payload decoded by the residual network.
3. FP16 payload, fused quantize, lookup and residual kernel.
4. Sparse cache over visited cells with temporal reuse.
5. Fused TF32/FP16 teacher for a stronger baseline.
6. Native Linux run with GPU activity tracing.
