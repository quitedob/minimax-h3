# 研究:RTX 50 原生 NVFP4 支持 + 是否把 H3 TE 换回 nvfp4_awq

**日期**: 2026-08-13
**研究方式**: Perplexity pro(1 主搜索 + 1 follow-up)+ 本地文件/日志实证
**问题**: RTX 5060 Ti(sm_120)是否原生支持 nvfp4?该不该把 H3 的 Qwen3-VL-32B 文本编码器从 int8_convrot 换回 nvfp4_awq?

---

## 关键发现(按置信度)

### [高] RTX 50 消费级 Blackwell 原生支持 FP4/NVFP4
- NVFP4 是 Blackwell 原生 4-bit 格式,由 RTX 50 系 tensor core 直接执行,不是 bf16 回退。
- 社区/NVIDIA 侧实测:**NVFP4 吞吐 ≈ FP8 的 2×**(发布测试里常见"roughly doubles");真实收益取决于负载。
- 关键配套:**ComfyUI 的 NVFP4 优化路径要求 CUDA 13.0**——本机正是 cu130(CUDA 13),满足。
- 来源:NVIDIA NVFP4 intro、TensorRT FP4 image-gen on RTX 50、ComfyUI 新优化博客(见 Sources)。

### [高] 本机 CUDA backend 已声明 nvfp4 为原生可用(本地实证)
- 本机所有 run 日志 `Native ops:` 均含 `nvfp4`(`comfyui_run*.log`)。
- `comfy_kitchen backend cuda: available=True`,capabilities 含 **`gemv_awq_w4a16`、`dequantize_nvfp4`、`scaled_mm_nvfp4`、`quantize_nvfp4`** → ComfyUI 的 nvfp4 执行路径在本机走 CUDA backend 原生路径,非 eager/bf16 回退。
- 即:**"RTX50 支持 nvfp4 native"在你的机器上不是纸面,comfy-kitchen 已经认可了。**

### [中高] nvfp4_awq 是 RTX 50 上 H3 TE 的主推/原生路径
- Comfy-Org / LilCheaty(MiniMax-H3-NVFP4)README 把 nvfp4 作为 Blackwell 上的 NVFP4 变体;int8_convrot 更多是"替代/基线"讨论。
- 质量侧:Qwen3-32B 在 NVFP4 上 ~2x 吞吐、GPQA/ARC 无可测精度损失(JarvisLabs 基准)——但这是 LLM 通用基准,**H3 TE 专属的 nvfp4 vs int8 质量对比未见公开数据**。

### [中高] 历史 nvfp4 离群值大概率是下载损坏,不是量化伪影
- 项目历史(`docs.md` / `research-h3-garbage.md`):nvfp4 TE 首次下载损坏(权重区域全零)→ 修复/重下,反量化逐层验证 MAE 0.006 正确,但 TE 输出有固定离群值(max=15974)。
- 同一时期 **int8 TE 也反复中招损坏**,且乱码根因最终被定位为**下载损坏**(aria2"满大小但区域清零")。→ 离群值更可能来自损坏文件/修复残留,而非 nvfp4 本身。
- 在原生 FP4 的 Blackwell 上重试干净的 nvfp4 是合理的(前提:文件完整性验证通过)。

### [低] 具体 sm_120 FP4 TFLOPS 数字 / H3 TE 专属基准
- 未检索到 RTX 5060 Ti 精确 FP4 TFLOPS 或 H3 TE 专属 nvfp4 vs int8 对比;社区建议"workload-dependent,自己本地基准"。

---

## 本机现状(关键)

- **nvfp4_awq TE 文件已在本机**:`F:\python\h3\models\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`(15.68GB,2026-08-13 18:38)。**无需重新下载。**
- **完整性已验证(2026-08-13)**:逐张量零区扫描 **2054 张量 / 15.69GB,0 个损坏** → `RESULT: OK`。历史上最大的下载损坏坑本次不存在。
- 当前生效 TE:int8_convrot(27.14GB,在 `text_encoders/`,已验证可用)。
- nvfp4 文件在 `models/`(暂存),未放入 `text_encoders/`。

---

## 项目文档审查

| 文档/日志 | 状态 | 相关结论 |
|---|---|---|
| `comfyui_run*.log` | RELIABLE(本地实证) | `Native ops` 含 nvfp4;comfy_kitchen CUDA backend 声明 `scaled_mm_nvfp4`/`gemv_awq_w4a16` |
| `docs.md` 阶段 5 | RELIABLE | int8_convrot TE 验证可用;nvfp4 曾损坏+离群值 |
| `research-h3-garbage.md` | RELIABLE(历史) | 乱码根因=下载损坏;nvfp4 反量化 MAE 0.006 正确 |
| `docs/research-h3-te-acceleration.md` | RELIABLE(本会话) | TE 单次运行、对采样贡献≈0 |

## 来源(Perplexity 提供,未逐条在线复核)

- NVIDIA NVFP4 intro: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- TensorRT FP4 on RTX 50: https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/
- ComfyUI NVFP4 优化: https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia
- LilCheaty/MiniMax-H3-NVFP4: https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4/blob/main/README.md
- Qwen3-32B NVFP4 基准(JarvisLabs): https://jarvislabs.ai/blog/nvfp4-rtxpro-6000
- FP4 概览: https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/

## 矛盾与缺口

- RTX 5060 Ti 精确 FP4 TFLOPS、H3 TE 专属 nvfp4 vs int8 质量对比:**无公开数据**(需要本地实测)。
- "nvfp4 TE 需 CUDA 13 否则 bf16 回退"的说法:**来源不一致**,未证实——但本机是 cu130,不受影响。

## 实测结果(2026-08-13 本机,RTX 5060 Ti)

### TE 加载 + 单次编码(同会话受控单测,CLIPLoader→MiniMaxH3ImageToVideo→PreviewAny)
| TE | staged 显存 | 加载+编码 |
|---|---|---|
| **nvfp4_awq** | **14956 MB** | **5.79 s** |
| int8_convrot | 25882 MB | **256.76 s** |

- **nvfp4 快 ~44×、省 ~11GB 显存**。int8 的 256s 还没算 VAE(已缓存),nvfp4 的 5.79s 反而含 VAE → 真实差距更大。
- int8_convrot 的慢主要来自 **27GB 权重 + convrot 去量化(CPU 侧,阻塞执行)**;nvfp4 的 gemv_awq_w4a16 路径轻得多。这解释了本项目历史上"模型初始化 3.5-4 分钟"且每次改配置重启都肉疼的原因——**int8 TE 加载就是大头之一**。
- 注:全流程里 fl2va(20GB fp8)加载仍占 init 大头(~3:50);TE 部分从 ~256s 降到 ~6s 是实打实的节省。

### 端到端 20 步(864×480/124帧,fp8 fl2va + nvfp4 TE)
- 总耗时 **00:12:57**(init ~3:56 + 采样 5:15 @15.6s/步 + 解码 ~3.5min)。
- **质量健康**:std 49.9-64.8、唯一色 41.9k-52.9k、暖调 R>G>B、运动随间隔递增(10.5→18.9→28.0)→ **无离群值、无质量回退**。产物 `MiniMax_H3_nvfp4te_20step_00001_.mp4`。
- **历史上 nvfp4 的固定离群值(max=15974)未复现** → 进一步确认那是旧文件损坏残留,不是 nvfp4 量化伪影。

## 建议(可操作)

1. **已确认可用:换 nvfp4 TE**(文件已拷入 `text_encoders/`,完整性 0 损坏,实测质量健康)。收益:TE 加载从 ~256s→~6s、省 11GB 显存。
2. **已落地文件**:`ComfyUI_windows_portable/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`(从 `models/` 拷贝);工作流切 CLIPLoader 到 nvfp4 文件即可。
3. **适用场景**:冷启动/改配置重启(每次省 ~4 分钟 init)、多段链条(每段重编码)、16GB 显存紧张时。
4. **若想极致**:SolAttn(采样)+ nvfp4 TE(init)叠加,是当前 H3 在本机的最优组合。

## 搜索覆盖

- **Perplexity**:1 主搜索(4 项清单)+ 1 follow-up(3 项硬缺口)——确认 RTX50 原生 FP4、ComfyUI NVFP4 需 CUDA13、离群值大概率是损坏。
- **本地**:模型文件清单(models/ 现有 nvfp4 文件)、comfy_kitchen CUDA backend 原生 ops 日志、项目历史文档。
- **未覆盖**:RTX 5060 Ti 精确 FP4 数字;H3 TE 专属 nvfp4 质量基准;nvfp4 文件完整性(待扫描)。
