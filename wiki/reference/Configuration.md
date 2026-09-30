# Configuration

Configurations are TOML files in `configs/`. Missing sections take defaults, so
`quality.toml` and `speed.toml` only override `[cache]`.

```toml
[render]
width = 1920
height = 1080
frames = 500
sequence = "camera"

[teacher]
hidden = [128, 128, 128, 64]
dtype = "f32"
seed = 1337

[cache]
position_grid = [24, 24, 24]
view_bins = 24
light_bins = 12
dtype = "f32"
interpolation = "nearest"

[benchmark]
warmup_frames = 50
measured_frames = 500
cuda_graph = true
max_vram_gib = 10.0
```

## Provided configurations

| File | Grid | View bins | Light bins | Cache size |
|---|---|---:|---:|---:|
| `speed.toml` | 16^3 | 8 | 4 | 12 MB |
| `default.toml` | 24^3 | 24 | 12 | 364.5 MB |
| `quality.toml` | 32^3 | 32 | 16 | 1536 MB |

Cache size is `6 x X x Y x Z x V x L x 16 bytes`. The runner prints the
estimated working set and refuses to start above `max_vram_gib` or the free
VRAM.

## Constraints

- Pixel count must be divisible by 32.
- Teacher layout is fixed to `[128, 128, 128, 64]`, FP32.
- Cache payload is FP32; the lookup mode is selected per benchmark mode, not by
  `interpolation`.
- Cache entries must fit i32 gather addressing.
