"""
Standalone SageAttention3 crash repro for the H3 long-sequence GPU-lost bug.

Usage (run in the Sage3 venv; crash poisons the CUDA context, so run
per_block_mean True and False as SEPARATE processes):

    # DS path under test (True) vs. not (False)
    venv/Scripts/python.exe comfyui_download/sage3_repro.py true 100
    venv/Scripts/python.exe comfyui_download/sage3_repro.py false 100

    # localize the bad launch (CUDA_LAUNCH_BLOCKING=1) / OOB/race/descriptor:
    set CUDA_LAUNCH_BLOCKING=1
    compute-sanitizer --tool memcheck venv/Scripts/python.exe comfyui_download/sage3_repro.py true 5

Devlog: docs/devlog.md s16.10. Detects the intermittent unknown-error / GPU-lost
fingerprint at (1,56,30509,128) bf16. Use per_block_mean False to separate the
DS TMA (delta_s / #382) path from the rest.
"""
import sys
import torch

from sageattn3.api import sageattn3_blackwell


def main() -> int:
    pb = (sys.argv[1].lower() in {"true", "1", "t"}) if len(sys.argv) > 1 else True
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    B, H, L = 1, 56, 30509
    dev = "cuda"
    print(f"per_block_mean={pb} iters={iters} shape=({B},{H},{L},128)", flush=True)
    torch.cuda.synchronize()
    q = torch.randn(B, H, L, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for i in range(iters):
        out = sageattn3_blackwell(q, k, v, is_causal=False, per_block_mean=pb)
        torch.cuda.synchronize()
        if i % 10 == 0:
            print(f"iter {i}: finite={torch.isfinite(out).all().item()}", flush=True)
        del out
    print(f"DONE per_block_mean={pb} iters={iters}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
