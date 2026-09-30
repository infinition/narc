# Capture Guide

Recording a sequence from a D3D12 title with RenderDoc, using The Elder Scrolls
IV: Oblivion Remastered (Unreal Engine 5) as the reference case. Use single-player
titles without anti-cheat only.

The export script (`scripts/renderdoc/narc_renderdoc_export.py`) has not been
validated on a real title yet. Start with one probe capture.

## Game settings

| Setting | Value | Reason |
|---|---|---|
| Resolution | 1920x1080, native | depth and color at the same size |
| Upscaling | off | |
| Frame generation | off | generated frames have no depth or camera |
| Motion blur, depth of field, grain, chromatic aberration, vignette | off | screen-space effects |
| Dynamic resolution | off | fixed size |
| Hardware ray tracing | default; software if RenderDoc crashes | |
| HUD | hidden (`tm` in the console) | |

Useful console commands: `tm` (hide HUD), `tfc` (free camera), `tai` (freeze
NPCs). Frame rate does not matter: only still frames are captured.

## 1. Probe capture

1. RenderDoc 1.3x or later, **Launch Application**, target the shipping
   executable (`...\OblivionRemastered\Binaries\Win64\OblivionRemastered-Win64-Shipping.exe`),
   or enable **Capture Child Processes** when starting through a launcher.
2. The connection tab must show `API: D3D12`. `API: None` means RenderDoc hooked
   the launcher, not the game.
3. Hide the HUD and press **F12** once. Save as `captures/<game>/rdc_probe/probe.rdc`.
4. Probe it:

   ```bash
   python scripts/renderdoc/narc_renderdoc_export.py probe captures/<game>/rdc_probe/probe.rdc captures/<game>/rdc_probe/probe.json
   ```

   `probe.json` lists depth candidates with the last event writing them, and
   the matrices and vectors found in the constant buffers at that event.
5. Fill `scripts/renderdoc/export_config.json` from
   `export_config.example.json`: depth resource name, constant buffer, matrix
   and translation variables, row-vector convention.

## 2. Convention check

Three captures during a slow move, then:

```bash
python scripts/renderdoc/narc_renderdoc_export.py export scripts/renderdoc/export_config.json captures/<game>/rdc_check captures/<game>/check
narc game validate-capture captures/<game>/check --pairs 2
```

`validate-capture` reprojects each frame from the previous one. A transposed
matrix, a missing translation or wrong depth turns the `matrix` line to `FAIL`.

## 3. Sequences

| Sequence | Content | `scene_class` |
|---|---|---|
| Static environment | camera only, NPCs frozen, stable weather | `static-cacheable` |
| Lighting | same path, changing time of day | `unknown` |
| Dynamic | NPCs, foliage, effects | `dynamic-unsafe` |

About 40 captures per sequence, one every 0.5 s, 2 to 5 degrees of rotation
between captures. A single `.rdc` is 0.5 to 2 GB. With `--every 8`, 5 frames
are baked and 35 stay unseen. A burst of 8 to 16 consecutive frames (RenderDoc
*Num Frames*) adds data for the temporal metric.

## Layout

```text
captures/<game>/
  rdc_probe/probe.rdc, probe.json
  rdc_static_01/capture_0001.rdc ...
  static_01/            exported, see Capture Format
```

`captures/` is ignored by git.

## Run

```bash
narc game validate-capture captures/<game>/static_01
narc game run captures/<game>/static_01 --every 8
```
