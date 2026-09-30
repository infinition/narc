"""Convert RenderDoc captures (.rdc) of a D3D12 game to the NARC capture format.

Experimental: not yet validated on a real title. Probe one capture first to
find the depth target and camera constant buffer, fill export_config.json,
then export and run `narc game validate-capture`.

  python narc_renderdoc_export.py probe  <file.rdc> <probe.json>
  python narc_renderdoc_export.py export <export_config.json> <rdc_dir> <out_dir>

Requires the Python version RenderDoc's pymodules were built for.
"""
import json
import os
import sys

import renderdoc as rd

FORMAT, VERSION = "narc-capture", 1


def open_capture(path):
    cap = rd.OpenCaptureFile()
    result = cap.OpenFile(path, "", None)
    ok = getattr(rd, "ResultCode", getattr(rd, "ReplayStatus", None)).Succeeded
    if result != ok:
        raise RuntimeError("cannot open %s: %s" % (path, result))
    if not cap.LocalReplaySupport():
        raise RuntimeError("capture %s cannot be replayed on this machine" % path)
    result, controller = cap.OpenCapture(rd.ReplayOptions(), None)
    if result != ok:
        raise RuntimeError("cannot replay %s: %s" % (path, result))
    return cap, controller


def actions(controller):
    roots = controller.GetRootActions() if hasattr(controller, "GetRootActions") else controller.GetDrawcalls()
    stack = list(reversed(roots))
    while stack:
        a = stack.pop()
        yield a
        stack.extend(reversed(a.children))


def names(controller):
    return {str(r.resourceId): r.name for r in controller.GetResources()}


def last_present(controller):
    present = None
    for a in actions(controller):
        if a.flags & rd.ActionFlags.Present:
            present = a
    return present


def last_depth_write(controller, rid):
    event = None
    for a in actions(controller):
        if str(a.depthOut) == str(rid):
            event = a.eventId
    return event


def save_texture(controller, rid, event, path, file_type):
    controller.SetFrameEvent(event, True)
    ts = rd.TextureSave()
    ts.resourceId = rid
    ts.mip = 0
    ts.slice.sliceIndex = 0
    ts.alpha = rd.AlphaMapping.Discard
    ts.destType = file_type
    controller.SaveTexture(ts, path)


def flatten(var, prefix=""):
    name = prefix + var.name
    if len(var.members):
        for m in var.members:
            for item in flatten(m, name + "."):
                yield item
    else:
        yield name, var.rows, var.columns, list(var.value.f32v[: var.rows * var.columns])


def cbuffer_variables(controller, event, stage):
    """All constant buffer variables bound to `stage` at `event`: {block: {name: (rows, cols, values)}}."""
    controller.SetFrameEvent(event, True)
    state = controller.GetPipelineState()
    refl = state.GetShaderReflection(stage)
    out = {}
    if refl is None:
        return out
    pipe = state.GetGraphicsPipelineObject()
    entry = state.GetShaderEntryPoint(stage)
    for slot, block in enumerate(refl.constantBlocks):
        try:
            if hasattr(state, "GetConstantBuffer"):  # RenderDoc <= 1.32
                b = state.GetConstantBuffer(stage, slot, 0)
                res, off, size = b.resourceId, b.byteOffset, b.byteSize
            else:  # RenderDoc >= 1.33 descriptor API
                d = state.GetConstantBlock(stage, slot, 0).descriptor
                res, off, size = d.resource, d.byteOffset, d.byteSize
            values = controller.GetCBufferVariableContents(pipe, refl.resourceId, stage, entry, slot, res, off, size)
        except Exception as e:  # keep probing other blocks
            out[block.name or "slot%d" % slot] = {"error": str(e)}
            continue
        out[block.name or "slot%d" % slot] = {n: [r, c, v] for v_ in values for n, r, c, v in flatten(v_)}
    return out


def probe(rdc, out_json):
    cap, controller = open_capture(rdc)
    try:
        nm = names(controller)
        textures = controller.GetTextures()
        present = last_present(controller)
        swap = [t for t in textures if t.creationFlags & rd.TextureCategory.SwapBuffer]
        depth = [t for t in textures if t.creationFlags & rd.TextureCategory.DepthTarget]
        report = {"file": rdc, "present_event": present.eventId if present else None,
                  "swapchain": [{"id": str(t.resourceId), "name": nm.get(str(t.resourceId), ""), "size": [t.width, t.height], "format": t.format.Name()} for t in swap],
                  "depth_candidates": [], "cbuffers": {}}
        for t in sorted(depth, key=lambda t: -t.width * t.height):
            ev = last_depth_write(controller, t.resourceId)
            report["depth_candidates"].append({"id": str(t.resourceId), "name": nm.get(str(t.resourceId), ""), "size": [t.width, t.height], "format": t.format.Name(), "last_depth_write_event": ev})
        best = next((d for d in report["depth_candidates"] if d["last_depth_write_event"]), None)
        if best:
            for stage in (rd.ShaderStage.Vertex, rd.ShaderStage.Pixel):
                blocks = cbuffer_variables(controller, best["last_depth_write_event"], stage)
                # Keep only matrices and small vectors: that is where the camera lives.
                report["cbuffers"][str(stage)] = {b: ({n: v for n, v in vars_.items() if isinstance(v, list) and v[0] * v[1] in (3, 4, 16)} if "error" not in vars_ else vars_) for b, vars_ in blocks.items()}
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", out_json)
    finally:
        controller.Shutdown()
        cap.Shutdown()


def matmul(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def world_to_clip(vars_, cam):
    m = vars_[cam["matrix"]][2]
    rows = [m[r * 4:(r + 1) * 4] for r in range(4)]
    if cam.get("row_vector", True):  # HLSL/Unreal mul(v, M): transpose to column-vector convention
        rows = [[rows[c][r] for c in range(4)] for r in range(4)]
    t = [0.0, 0.0, 0.0]
    for name in cam.get("translation", []):
        v = vars_[name][2]
        s = cam.get("translation_scale", {}).get(name, 1.0)
        t = [t[k] + s * v[k] for k in range(3)]
    translate = [[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]
    return matmul(rows, translate)


def export(config_path, rdc_dir, out_dir):
    cfg = json.load(open(config_path))
    files = sorted(f for f in os.listdir(rdc_dir) if f.lower().endswith(".rdc"))
    os.makedirs(out_dir, exist_ok=True)
    frames, size = [], None
    for i, f in enumerate(files):
        cap, controller = open_capture(os.path.join(rdc_dir, f))
        try:
            nm = names(controller)
            by_name = {}
            for t in controller.GetTextures():
                by_name.setdefault(nm.get(str(t.resourceId), ""), t)
            present = last_present(controller)
            swap = next(t for t in controller.GetTextures() if t.creationFlags & rd.TextureCategory.SwapBuffer)
            depth = by_name[cfg["depth_resource_name"]]
            dev = last_depth_write(controller, depth.resourceId)
            d = os.path.join(out_dir, "frame_%06d" % i)
            os.makedirs(d, exist_ok=True)
            save_texture(controller, swap.resourceId, present.eventId, os.path.join(d, "color.png"), rd.FileType.PNG)
            save_texture(controller, depth.resourceId, dev, os.path.join(d, "depth.exr"), rd.FileType.EXR)
            cam_cfg = cfg["camera"]
            stage = getattr(rd.ShaderStage, cam_cfg.get("stage", "Vertex"))
            block = cbuffer_variables(controller, dev, stage)[cam_cfg["block"]]
            meta = {"index": i, "time_s": i * cfg.get("frame_interval_s", 0.5),
                    "camera": {"world_to_clip": world_to_clip(block, cam_cfg)},
                    "extra": {"rdc": f, "depth_event": dev, "present_event": present.eventId,
                              "raw": {k: block[k] for k in [cam_cfg["matrix"]] + cam_cfg.get("translation", [])}}}
            with open(os.path.join(d, "metadata.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            frames.append("frame_%06d" % i)
            size = (swap.width, swap.height)
            print("exported", f)
        finally:
            controller.Shutdown()
            cap.Shutdown()
    manifest = {"format": FORMAT, "version": VERSION, "name": cfg.get("name", os.path.basename(os.path.normpath(out_dir))),
                "source": "renderdoc", "width": size[0], "height": size[1],
                "depth": {"kind": "ndc", "sky_ndc": cfg.get("sky_ndc", 0.0)},
                "scene_class": cfg.get("scene_class", "unknown"), "frames": frames,
                "notes": ["Exported by scripts/renderdoc/narc_renderdoc_export.py", "camera config: %s" % json.dumps(cfg["camera"])]}
    with open(os.path.join(out_dir, "capture.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote", os.path.join(out_dir, "capture.json"))


if __name__ == "__main__":
    rd.InitialiseReplay(rd.GlobalEnvironment(), [])
    try:
        if len(sys.argv) == 4 and sys.argv[1] == "probe":
            probe(sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 5 and sys.argv[1] == "export":
            export(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print(__doc__)
            sys.exit(2)
    finally:
        rd.ShutdownReplay()
