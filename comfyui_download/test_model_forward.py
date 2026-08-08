#!/usr/bin/env python
"""Standalone test: load fl2va UNET and run one forward pass.
Determines whether the model produces reasonable output or garbage.
"""
import sys, os
C = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI'
sys.path.insert(0, C); os.chdir(C)
import torch
import comfy.sd
import comfy.model_management

UNET = r'F:\python\h3\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors'

print("Loading UNET ...")
model = comfy.sd.load_diffusion_model(UNET, model_options={})
print("UNET loaded:", type(model))

# A random video latent [B, 24, T, H, W] (H=30, W=54 for 864x480, T=37 for 124 frames)
x = torch.randn(1, 24, 37, 30, 54, device='cpu')
# Random context [B, T, 5120]
ctx = torch.randn(1, 11, 5120, device='cpu')
# A timestep (flow model uses sigma-based timestep)
ts = torch.tensor([0.5], device='cpu')

# Get model config / sampling to compute a proper timestep
model_config = model.model.model_config
print("model_config:", type(model_config))
try:
    sampling = model.get_model_object("model_sampling")
    print("model_sampling:", type(sampling))
except Exception as e:
    print("no sampling:", e)

try:
    with torch.no_grad():
        out = model.model(x, ts, ctx)
    print("forward OK")
    print("out shape:", tuple(out.shape))
    o = out.detach().float()
    print("out mean=%.6f std=%.6f min=%.6f max=%.6f" % (o.mean(), o.std(), o.min(), o.max()))
    print("NaN:", o.isnan().sum().item(), "Inf:", o.isinf().sum().item())
except Exception as e:
    import traceback
    traceback.print_exc()
print("Done.")
