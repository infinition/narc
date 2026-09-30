# Capture Format

```text
<capture>/
  capture.json          manifest
  frame_000000/
    color.png           final LDR color, sRGB, no HUD            (required)
    depth.exr           raw NDC depth in channel R, float 32     (level 1)
    normal.exr          world-space normal in RGB                (level 2)
    motion.exr          screen motion in pixels in RG            (level 3)
    gbuffer/            extra buffers                            (level 4)
    metadata.json       index, time, camera
```

The level is detected from the files present on every frame.

## capture.json

```json
{
  "format": "narc-capture",
  "version": 1,
  "name": "oblivion-static-01",
  "source": "renderdoc",
  "width": 1920,
  "height": 1080,
  "depth": { "kind": "ndc", "sky_ndc": 0.0 },
  "scene_class": "static-cacheable",
  "frames": ["frame_000000", "frame_000001"],
  "notes": []
}
```

| Field | Meaning |
|---|---|
| `depth.kind` | Only `ndc`: the raw depth buffer value |
| `depth.sky_ndc` | Cleared depth: `0.0` for reversed Z (Unreal Engine 5), `1.0` for standard Z, `null` if unknown |
| `scene_class` | `static-cacheable`, `dynamic-unsafe` or `unknown` |
| `frames` | Frame directories in temporal order |

## metadata.json

```json
{
  "index": 0,
  "time_s": 0.0,
  "camera": {
    "world_to_clip": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0.1], [0, 0, 1, 0]],
    "position": [0.0, 0.0, 0.0]
  },
  "tags": [],
  "extra": {}
}
```

- `world_to_clip`: row by row, column-vector convention (`clip = M * [x y z 1]`),
  in a world frame shared by all frames. It must be the matrix used to rasterize
  the depth buffer.
- `position`: optional, derived from the matrix.
- Image convention is Direct3D: NDC y points up, row 0 is the top row.
- `extra`: free-form, kept for audit.

A camera plus raw NDC depth is enough to unproject with any convention:
handedness, reversed Z, infinite far plane.
