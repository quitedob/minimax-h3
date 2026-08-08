#!/usr/bin/env python
"""In-place patch comfy_quant tensors in ComfyUI-format H3 model files.

The Comfy-Org/MiniMax-H3 files ship comfy_quant tensors as null byte
placeholders. ComfyUI's quantized-weight loader expects a JSON config there
({"format": "int8_tensorwise"/"nvfp4", ...}) and crashes with
"utf-32-be codec can't decode" on the nulls.

This script overwrites each comfy_quant tensor IN PLACE with the correct
JSON (padded with spaces to the tensor's exact byte length — trailing
whitespace is valid JSON), so no file rewrite / extra disk is needed.
"""
import json
import struct
import sys

# Quant configs per file variant. convrot_groupsize 256 = ComfyUI default.
CONFIGS = {
    # fl2va / ref2va int8 transformer: per-row int8 + convrot rotation
    "fl2va": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    # qwen3vl text encoder, NVFP4 AWQ layers (int4-packed uint8 weights)
    "nvfp4": {"format": "nvfp4"},
    # text encoder embed_tokens: int8 per-row (no convrot)
    "embed_int8": {"format": "int8_tensorwise"},
}


def patch_file(path, conf_for_tensor):
    """conf_for_tensor(name) -> dict (JSON to write) or None to skip."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

    patched = 0
    skipped = 0
    for name, meta in header.items():
        if not name.endswith("comfy_quant"):
            continue
        conf = conf_for_tensor(name)
        if conf is None:
            skipped += 1
            continue
        begin, end = meta["data_offsets"]
        tlen = end - begin
        payload = json.dumps(conf, separators=(",", ":")).encode("utf-8")
        if len(payload) > tlen:
            print(f"  [FAIL] {name}: json {len(payload)}B > tensor {tlen}B")
            return 1
        payload = payload + b" " * (tlen - len(payload))  # pad w/ trailing spaces
        abs_pos = data_start + begin
        with open(path, "r+b") as f:
            f.seek(abs_pos)
            f.write(payload)
        patched += 1
    print(f"  patched {patched} comfy_quant tensors, skipped {skipped}")
    return 0


def verify(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        bad = 0
        for k in f.keys():
            if not k.endswith("comfy_quant"):
                continue
            try:
                json.loads(f.get_tensor(k).numpy().tobytes())
            except Exception as e:
                bad += 1
                print(f"  [VERIFY FAIL] {k}: {e}")
        return bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fl2va"
    path = sys.argv[2]
    # optional: convrot <true|false> <groupsize>, e.g. fl2va <path> false 256
    convrot = sys.argv[3] if len(sys.argv) > 3 else None
    groupsize = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    if mode == "fl2va":
        conf = dict(CONFIGS["fl2va"])
        if convrot is not None:
            conf["convrot"] = convrot.lower() in ("true", "1", "yes")
            conf["convrot_groupsize"] = groupsize
        cfg = lambda n: conf
    elif mode == "text":
        def cfg(n):
            if "embed_tokens" in n:
                return CONFIGS["embed_int8"]
            return CONFIGS["nvfp4"]
    else:
        sys.exit("unknown mode")
    print(f"Patching {path} ...")
    rc = patch_file(path, cfg)
    if rc:
        sys.exit(rc)
    bad = verify(path)
    print(f"verify: {bad} failures")
    sys.exit(0 if bad == 0 else 1)
