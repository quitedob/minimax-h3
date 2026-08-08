#!/usr/bin/env python
"""Isolate-test the MiniMax H3 text encoder (qwen3vl nvfp4 awq).

Loads the CLIP the same way CLIPLoader(type="minimax") does, tokenizes a
prompt, encodes it, and prints stats of the conditioning so we can judge
whether the encoder produces sensible output or garbage.
"""
import sys, os
C = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI'
sys.path.insert(0, C)
os.chdir(C)

import torch
import folder_paths
import comfy.sd
import comfy.model_management
from comfy.sd import CLIPType

ENC = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'

def show(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().float()
        print(f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
              f"mean={t.mean():.4f} std={t.std():.4f} min={t.min():.4f} max={t.max():.4f} "
              f"nan={t.isnan().sum().item()} inf={t.isinf().sum().item()}")
    else:
        print(f"{name}: {type(t)}")

print("Loading CLIP ...")
clip = comfy.sd.load_clip(ckpt_paths=[ENC], clip_type=CLIPType.MINIMAX, model_options={})
print("CLIP loaded:", type(clip))

for prompt in ["A golden retriever puppy running in a park at sunset",
               "a red sports car on a mountain road"]:
    print(f"\n=== prompt: {prompt[:40]} ===")
    tokens = clip.tokenize(prompt)
    print("tokens type:", type(tokens))
    cond = clip.encode_from_tokens_scheduled(tokens)
    print("cond type:", type(cond))
    if isinstance(cond, dict):
        for k, v in cond.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    show(f"cond['{k}']['{k2}']", v2)
            else:
                show(f"cond['{k}']", v)
    else:
        show("cond", cond)

# Compare: two different prompts should give clearly different embeddings.
print("\nDone.")
