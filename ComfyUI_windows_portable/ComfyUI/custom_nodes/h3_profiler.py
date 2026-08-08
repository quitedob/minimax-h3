"""H3 sample profiler: wraps guider.sample in torch.profiler, dumps trace + per-kernel summary.

Drop-in replacement for SamplerCustomAdvanced (same inputs/outputs). Run a 1-step
workflow to get a per-kernel CUDA breakdown of the H3 sampling forward.
"""
import json
import torch
import comfy.sample

OUT_SUMMARY = r'F:\python\h3\comfyui_download\profile_summary.json'
OUT_TRACE = r'F:\python\h3\comfyui_download\profile_trace.json'


class H3SampleProfiler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "noise": ("NOISE",),
            "guider": ("GUIDER",),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom"

    def execute(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))
        latent["samples"] = latent_image
        noise_mask = latent.get("noise_mask")

        from torch.profiler import profile, ProfilerActivity
        samples = None
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=False, profile_memory=False, with_stack=False) as prof:
                samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas,
                                        denoise_mask=noise_mask, callback=None, disable_pbar=True, seed=noise.seed)
                torch.cuda.synchronize()
            try:
                prof.export_chrome_trace(OUT_TRACE)
            except Exception as e:
                print("[H3Profiler] trace export failed:", repr(e))
            rows = []
            for e in prof.key_averages():
                self_cu = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0
                tot_cu = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0) or 0
                self_cpu = getattr(e, "self_cpu_time_total", None) or getattr(e, "self_time_total", 0) or 0
                rows.append({
                    "op": e.key, "self_cuda_ms": round(self_cu / 1000, 2),
                    "total_cuda_ms": round(tot_cu / 1000, 2),
                    "self_cpu_ms": round(self_cpu / 1000, 2), "count": e.count,
                })
            rows.sort(key=lambda r: -r["self_cuda_ms"])
            with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)
            print("\n=== H3 PROFILE TOP CUDA SELF-TIME (ms) ===")
            for r in rows[:50]:
                print(f"{r['self_cuda_ms']:9.1f}ms x{r['count']:4d}  {r['op']}")
            print("\n=== H3 PROFILE TOP CPU SELF-TIME (ms, offload/copy proxy) ===")
            for r in sorted(rows, key=lambda r: -r["self_cpu_ms"])[:15]:
                print(f"{r['self_cpu_ms']:9.1f}ms x{r['count']:4d}  {r['op']}")
            print(f"[H3Profiler] {len(rows)} rows -> {OUT_SUMMARY}")
        except Exception as e:
            import traceback
            print("[H3Profiler] profiling failed, running without profiler:", repr(e))
            traceback.print_exc()
            samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas,
                                    denoise_mask=noise_mask, callback=None, disable_pbar=True, seed=noise.seed)

        samples = samples.to(comfy.model_management.intermediate_device())
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return (out, out)


NODE_CLASS_MAPPINGS = {"H3SampleProfiler": H3SampleProfiler}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SampleProfiler": "H3 Sample Profiler"}
