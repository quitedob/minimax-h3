#!/usr/bin/env python
"""DECISIVE diagnostic: does conditioning actually affect the H3 DiT?

1) Load int8 text encoder, encode the A/B prompt, print conditioning stats
   (does it still have the fixed max~15974 outlier the nvfp4 encoder had?)
2) Run the fl2va DiT forward TWICE with the SAME noise/timestep:
   once with real conditioning, once with a dummy (zero) conditioning.
   If both outputs are (nearly) identical => the DiT is ignoring conditioning.
"""
import sys, os
C = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI'
sys.path.insert(0, C); os.chdir(C)
import torch
import comfy.sd
from comfy.sd import CLIPType

TE = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
UNET = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors'
PROMPT = "A cinematic shot of a golden retriever puppy running through a sunlit park at golden hour"

def stats(name, t):
    t = t.detach().float()
    print(f"  {name}: shape={tuple(t.shape)} mean={t.mean():.4f} std={t.std():.4f} "
          f"min={t.min():.4f} max={t.max():.4f} nan={t.isnan().sum().item()} inf={t.isinf().sum().item()}")

print("Loading int8 text encoder ...")
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=CLIPType.MINIMAX, model_options={})
cond = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
print("cond len:", len(cond))
t0, meta = cond[0]
stats("conditioning[0]", t0)

# Also compare two DIFFERENT prompts through int8 encoder
cond2 = clip.encode_from_tokens_scheduled(clip.tokenize("a red sports car on a mountain road at night"))
t2, _ = cond2[0]
stats("conditioning[1] (other prompt)", t2)
a = t0.detach().float(); b = t2.detach().float()
a_n = a/(a.norm(dim=-1,keepdim=True)+1e-8); b_n = b/(b.norm(dim=-1,keepdim=True)+1e-8)
print("  cosine two prompts:", (a_n*b_n).sum(-1).mean().item())

print("Loading fl2va DiT ...")
model = comfy.sd.load_diffusion_model(UNET, model_options={})
print("UNET loaded")

torch.manual_seed(42)
x = torch.randn(1, 24, 37, 30, 54, device='cpu')
ts = torch.tensor([0.5], device='cpu')

# Real conditioning: broadcast t0 to expected ctx shape [1, T, 5120]
ctx_real = t0.detach().float()
if ctx_real.dim() == 2:
    ctx_real = ctx_real.unsqueeze(0)
print("ctx_real shape:", tuple(ctx_real.shape))

with torch.no_grad():
    out_real = model.model(x, ts, ctx_real)
    print("forward (real cond) OK")

    ctx_zero = torch.zeros_like(ctx_real)
    out_zero = model.model(x, ts, ctx_zero)
    print("forward (zero cond) OK")

or_ = out_real.detach().float(); oz = out_zero.detach().float()
diff = (or_ - oz).abs()
print("\n=== CONDITIONING EFFECT ===")
print(f"out(real) mean={or_.mean():.6f} std={or_.std():.6f}")
print(f"out(zero) mean={oz.mean():.6f} std={oz.std():.6f}")
print(f"mean-abs-diff={diff.mean():.6f} max-diff={diff.max():.6f}")
print(f"relative change={(diff.mean()/(or_.std()+1e-8)).item():.6f}")
print(f"max-relative={(diff.max()/(or_.std()+1e-8)).item():.6f}")
print("=> CONDITIONING IGNORED" if diff.mean() < 0.01 else "=> CONDITIONING HAS EFFECT")
