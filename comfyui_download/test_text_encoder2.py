#!/usr/bin/env python
"""Inspect MiniMax H3 text encoder conditioning (list structure)."""
import sys, os
C = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI'
sys.path.insert(0, C); os.chdir(C)
import torch
import comfy.sd
from comfy.sd import CLIPType

ENC = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'

def stats(name, t):
    t = t.detach().float()
    print(f"  {name}: shape={tuple(t.shape)} mean={t.mean():.4f} std={t.std():.4f} "
          f"min={t.min():.4f} max={t.max():.4f} nan={t.isnan().sum().item()} inf={t.isinf().sum().item()}")

clip = comfy.sd.load_clip(ckpt_paths=[ENC], clip_type=CLIPType.MINIMAX, model_options={})

embeddings = []
for prompt in ["A golden retriever puppy running in a park at sunset",
               "a red sports car on a mountain road at night"]:
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    print(f"\nprompt: {prompt[:40]}")
    print("cond type:", type(cond), "len:", len(cond))
    item = cond[0]
    print("item type:", type(item), "len:", len(item) if hasattr(item,'__len__') else '-')
    t0, meta = item
    print("t0 type:", type(t0))
    stats("cond tensor", t0)
    print("meta keys:", list(meta.keys()) if isinstance(meta, dict) else meta)
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor):
                        stats(f"meta[{k}][{k2}]", v2)
    embeddings.append(t0)

if len(embeddings) == 2:
    a, b = embeddings[0].detach().float(), embeddings[1].detach().float()
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    cos = (a_n * b_n).sum(-1)
    print(f"\n两个不同prompt的cosine相似度: mean={cos.mean():.4f} (0-1, 明显不同应显著<1)")
