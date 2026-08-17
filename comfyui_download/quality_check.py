#!/usr/bin/env python
"""Frame-level quality check for an H3 output mp4: std, unique colors, mean RGB, motion.
Healthy reference (project docs): std 34-65, >700 unique colors, warm R>G>B, motion rises with gap.
"""
import sys, cv2, numpy as np

def check(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("FAIL: cannot open", path); return
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    print(f"frames={len(frames)}")
    for idx in [10, 60, 120]:
        if idx >= len(frames): continue
        f = frames[idx].astype(np.float32)
        print(f"frame {idx}: std={f.std():.1f}  unique_colors={len(np.unique(frames[idx].reshape(-1,3), axis=0))}  "
              f"mean_rgb=({f[:,:,2].mean():.0f},{f[:,:,1].mean():.0f},{f[:,:,0].mean():.0f})")
    # motion: mean abs diff between frames at increasing gaps
    for gap in [1, 5, 20]:
        if len(frames) <= gap: continue
        a = frames[gap].astype(np.float32); b = frames[0].astype(np.float32)
        print(f"motion gap={gap}: mean_abs_diff={np.abs(a-b).mean():.2f}/255")

if __name__ == "__main__":
    check(sys.argv[1])
