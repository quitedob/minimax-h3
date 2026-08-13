# MiniMax-H3 本地部署全记录

**日期**: 2026-08-03
**目标**: 从 ModelScope 下载 MiniMax-H3 并在本机运行视频生成流程
**硬件**: NVIDIA RTX 5060 Ti 16GB 显存 / 64GB 内存 / CUDA 12.8 / Windows 11 / Python 3.12 / F: 盘

---

## 一、硬件指标(实测)

### GPU 峰值(生成过程中 nvidia-smi 实测)
| 指标 | 值 |
|---|---|
| GPU 型号 | NVIDIA GeForce RTX 5060 Ti |
| 显存总量 | 16311 MiB (16 GB) |
| **峰值显存占用** | **14552 MiB (≈14.2 GB, 87%)** |
| GPU 利用率 | 100% (采样阶段) |
| torch 版本 | 2.13.0+cu130 (ComfyUI 便携版内置) |

### 内存 (RAM) 峰值(推算)
| 组件 | 驻留内存 |
|---|---|
| fl2va 扩散模型 (int8 反量化后 bf16) | ~20 GB |
| text_encoder (nvfp4 AWQ) | ~15 GB |
| video_vae (fp16) | ~5 GB |
| audio_vae (fp32) | ~0.6 GB |
| ComfyUI + torch 运行开销 | ~5-8 GB |
| **合计(模型加载峰值)** | **≈40-48 GB**(64GB 内可运行) |

> 说明:16GB 显存装不下 ~40GB 模型,ComfyUI 使用**动态显存加载**(层在 RAM↔VRAM 间流动),采样时显存约 87% 占用。

### 生成速度(480p)
- 5 秒视频(124 帧):**约 7-9 分钟**
  - 初始化(模型加载+JIT):~3 分 30 秒
  - 采样:20 步 × 12.7 秒/步 ≈ 4 分 15 秒(10 步版 ≈ 2 分钟)
  - 解码:约 1-2 分钟

---

## 二、关键结论(研究阶段)

1. **MiniMax-H3 架构**: 33B Omni-Transformer(含 13B 可跳过的 AdaLN)+ Qwen3-VL-32B 文本编码器 + 视觉/音频 VAE。BF16 全量单变体 ≈134GB。
2. **本机可行性**: BF16 需要 ~130GB 内存,本机 64GB 跑不了 → **必须用量化版本**。
3. **量化方案**: ComfyUI 官方推荐 `Comfy-Org/MiniMax-H3` 的 **pruned-int8 权重集**(共 ~42GB):
   - `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (19.53 GiB)
   - `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (14.61 GiB)
   - `minimax_h3_video_vae_fp16.safetensors` (4.85 GiB)
   - `minimax_h3_audio_vae_fp32.safetensors` (0.56 GiB)
4. **ComfyUI 版本**: v0.30.0+ 内置原生 H3 节点(MiniMaxH3ImageToVideo 等)。

---

## 三、操作时间线

### 阶段 1:环境准备
1. 检查环境:git 2.52 + git-lfs 3.7.1 已装,modelscope 未装(F: 剩 119GB)。
2. `GIT_LFS_SKIP_SMUDGE=1` 克隆 ModelScope 仓库获取结构+大小 → 确认单仓库含 FL2VA/Ref2VA,全仓 ~464GB。
3. `pip install modelscope`(1.39.0)。
4. 下载 ComfyUI 便携版 v0.30.0(2GB,7z 用已装的 7-Zip 解压到 `ComfyUI_windows_portable/`)。
5. 用 7-Zip 解压成功(5795 文件夹/56101 文件)。

### 阶段 2:模型下载(踩坑最多)
6. 从 ModelScope 用 modelscope CLI 下载 4 个文件 → **速率仅 3MB/s,太慢**。
7. 装 aria2(多线程),测试 → **8连接 5.3MB/s → 实际 20MB/s+**。
8. 切换到 aria2 下载,但**出现"文件满大小但数据块被清零"的损坏**(详见问题记录)。
9. 修复/重下经过:
   - nvfp4 text_encoder 首次下载损坏(权重区域全零)→ **重下 15.7GB**。
   - int8 text_encoder 下载 → **21% 数据被清零(5.8GB)** → 进行区域修复(见阶段 4)。

### 阶段 3:ComfyUI 配置与量化适配
10. ComfyUI 启动成功,torch 2.13.0+cu130,识别 GPU RTX 5060 Ti。
11. **关键问题**: Comfy-Org 模型的 `comfy_quant` 张量全是空字节(占位),ComfyUI 加载报 `utf-32-be codec can't decode`。
12. **解决方案**: 写 `patch_comfy_quant.py` **原位改写** comfy_quant 张量为正确的量化配置 JSON:
    - fl2va → `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`
    - text_encoder nvfp4 → `{"format":"nvfp4"}`,embed_tokens → `{"format":"int8_tensorwise"}`
13. **量化配置验证**(对比 bf16 真值,通过 HF range 请求):
    - fl2va 反量化:convrot g=256 → MAE **0.0006** ✓
    - text_encoder nvfp4 → MAE **0.006** ✓
14. ComfyUI 升级到 master 分支(模型文件预期版本)。

### 阶段 4:生成调试(当前进行中)
15. 构建 H3 T2V 工作流(原生节点):
    `UNETLoader → MiniMaxH3SigmaShift → MiniMaxH3ImageToVideo → BasicGuider + KSamplerSelect(res_multistep) + BasicScheduler(simple) → SamplerCustomAdvanced → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo`
16. **验证问题**: 管线跑通(输出合法 mp4),但**画面为灰色垃圾**(RGB~120 均匀)。
17. 排查: 模型权重全部验证正确、sigma 正常、conditioning 非零但有**固定离群值 max=15974**。
18. 决定换 int8_convrot text_encoder(与 fl2va 相同的已验证 int8 路径)→ 下载 25.28GB → **发现同样损坏(21% 清零)** → 正在区域修复。

---

## 四、遇到的问题与解决

| 问题 | 解决方案 |
|---|---|
| Perplexity MCP 连不上 | 配置代理 7897 端口;修复 `/tmp` 路径 bug(打补丁 client.py 用 tempfile) |
| ModelScope 下载慢(3MB/s) | 换 aria2 多线程(20MB/s+) |
| 下载文件"满大小但区域清零" | 对比 HF 字节验证;重下或用 numpy 扫描零区域 + range 下载修复 |
| comfy_quant 空字节导致加载崩溃 | 写脚本原位改写 comfy_quant 为正确 JSON |
| 量化配置不确定 | 用 bf16 真值做 MAE 对比,精确定位正确配置 |
| SaveLatent 不支持 NestedTensor | 改用 numpy 直接扫描文件 |
| 生成输出灰色 | **排查中**(当前聚焦 text_encoder 量化问题) |

---

## 五、磁盘占用(F: 盘)

- ComfyUI 便携版: ~4.4GB
- 模型: fl2va(19.5G)+ nvfp4 te(15G)+ int8 te(26G)+ 双VAE(5.4G)≈ **66GB**
- 剩余空间: 约 47GB(正在下载/修复中,数值会变化)

---

## 六、最终结论(截至 2026-08-03 19:47)

**结果: 所有生成的视频均为乱码/灰色内容,已删除。**

### 排查结论
1. **工作流连线正确**: 官方模板(视频_minimax_h3_t2v.json 的 "Image to Video (MiniMax H3)" 子图)的连线与手工构建的完全一致(UNETLoader + CLIPLoader + MiniMaxH3ImageToVideo + BasicGuider + res_multistep + simple + 20步 + SamplerCustomAdvanced + VAEDecode/Audio + CreateVideo)。官方模板不需要 MiniMaxH3SigmaShift(我方曾误加)。
2. **模型权重全部验证正确**:
   - fl2va int8 反量化 vs bf16: MAE 0.0006 ✓
   - text_encoder nvfp4 反量化 vs bf16: MAE 0.006 ✓
3. **根本原因未最终定位**。已知现象:
   - text_encoder 条件输出对两个不同 prompt 的 std(112.66)和 max(15974)完全相同——有固定离群值,疑似深层量化细节问题。
   - 输出帧为灰/噪声,模型未能产生有效内容。
4. **下载过程反复出现"文件满大小但区域被清零"的损坏**(nvfp4 和 int8 两个 text_encoder 都中招),修复/重下耗时巨大,可能是最终失败的诱因之一。

### 清理动作(2026-08-03 ~19:47)
- ✅ 删除全部生成视频(5 个 mp4,共 38MB)
- ✅ 关闭 llamacpp VLM 服务器
- ✅ 关闭 ComfyUI
- 保留: ComfyUI 便携版 + 模型文件(~66GB)在 `F:\python\h3\ComfyUI_windows_portable\`,供后续排查

### 建议
- 若继续: 优先排查 text_encoder 离群值来源,或等待 ComfyUI/Comfy-Org 更新 H3 支持修复。
- 或改用 MiniMax 云端 API(绕开本地量化问题)。

*文档更新至 2026-08-03 19:47*

---

## 阶段 5:重下真实模型,一次通过 ✅(2026-08-05 15:08)

### 关键动作
1. **重新下载全部 5 个模型**到 `F:\python\h3\models\`(真实权重,comfy_quant 元数据完整):
   - `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` (20.9GB,fl2va 换为 fp8_scaled 格式)
   - `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` (20.9GB,新增 ref2va)
   - `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27.1GB,int8 text encoder)
   - `minimax_h3_video_vae_fp16.safetensors` / `minimax_h3_audio_vae_fp32.safetensors`
2. **完整性验证**: 全量读盘零区扫描 → **5 个文件全部 0 个 ≥16MB 零区**,零字节占比 ≤1.2% → 无损坏。
3. **无需再 patch comfy_quant**: fl2va/ref2va 自带 `{"format":"float8_e4m3fn"}`,te 自带 `int8_tensorwise+convrot256`,加载零报错。
4. 移入 ComfyUI: fl2va/ref2va → `diffusion_models/`,te → `text_encoders/`,双 VAE → `vae/`。
5. 工作流改用 fp8_scaled unet(`comfyui_download/h3_fp8.json`),20 步 res_multistep + simple,864×480/124帧。

### 结果: 生成成功,输出为真实内容
- `ComfyUI_windows_portable/ComfyUI/output/video/MiniMax_H3_fp8_00001_.mp4`(864×480, 5.17s, 124 帧, 含 AAC 音频)。
- **像素验证非乱码**: 帧 std=40~65(旧乱码为均匀灰~120,std≈0);唯一色 700+;全动态范围;暖金色调(R>G>B,匹配 golden hour prompt)。
- **真实运动**: 相邻帧(0.04s)均差 ~10/255,间隔越大差值越大(达 44/255),每帧都在动。
- 音频非静音(mean −32.9dB)。

### 根因结论
**乱码根因是下载损坏**(aria2 多连接"满大小但区域清零"),而非量化配置或工作流问题。换成干净的真实权重 + fp8_scaled 后一次通过,总耗时 ~15.5 分钟(加载 ~4 分 + 采样 ~5 分 + 解码 ~6 分)。

*文档更新至 2026-08-05 15:10*

---

## 下载规范(强制,2026-08-05)

**禁止使用 curl / aria2 / 任何裸 HTTP 下载模型。** 一律用官方 Python 包(都已注册、凭证已缓存):

| 来源 | 客户端 | 环境 |
|---|---|---|
| ModelScope | `python -m modelscope download --model ...` (modelscope 1.39.0) | 系统 Python312 |
| HuggingFace | `hf_hub_download`/`snapshot_download` (huggingface_hub 1.25.1) | ComfyUI portable python (`ComfyUI_windows_portable\python_embeded\python.exe`) |

**原因**: aria2 多线程下载反复产生"满大小但区域清零"的损坏文件(text_encoder ×2、fl2va),是乱码输出的根本原因。

---

## 阶段 6:Sol-Attn 加速 H3 attention 成功 ✅(2026-08-05 19:05)

### 背景
- sage/sdpa 都动不了 H3 attention(`low_precision_attention=False` 锁死)。
- 发现 `kijai/ComfyUI-SolAttn_triton`(arXiv 2607.24027 Sol-Attn):**免训练稀疏注意力 + Morton Z-order token 重排 + INT8 QK + 精确-KV sink**,专为 MiniMax H3 设计(4090/5090 测试,sm_120 同类)。

### 卡点与修复(关键)
- 内核要求 triton 运行时编译 `cuda_utils` helper,但 **embedded python 没有 CPython 头文件**(`Include\` 只有 greenlet)和 `python313.lib` → tcc 编译报 `Python.h not found`。
- **修复**:
  1. 从 CPython 3.13.14 源码 tarball 提取 `Include/*.h` + `Include/cpython/`(59 个)+ `PC/pyconfig.h`(用系统 Python312 的真实 pyconfig.h 代替)→ 铺入 `python_embeded\Include\`
  2. 用 **MSVC**(`dumpbin /exports` + `lib.exe /def`)从 `python_embeded\python313.dll` 生成 `python313.lib` → 放入 `triton\backends\nvidia\lib\`
- 修复后 `CudaUtils` 编译通过,`sol_attn` / `sol_attn_int8` 内核在 sm_120 跑通。

### 实测结果
| 指标 | 基线 | SolAttn |
|---|---|---|
| attention 内核 | cudnn flash 7.7s/步 | `_forward_int8_ptr` ~4s/步(**↓55%**) |
| 每步墙钟 | ~12.7s | **~10.4s(↓18%)** |
| 10 步生成总耗时 | — | **210s**(含 20s 缓存初始化 + 解码) |

- 工作流:`comfyui_download/h3_solattn_gen.json`(SolAttnPatch 插在 UNETLoader→BasicGuider 之间,tau=1.3 / start 0.2 / end 0.9 / int8_qk / morton 2d_frame / sink exact_kv_and_rows)。
- **质量:视觉无损**(帧统计与基线一致:std 40-65、700+ 唯一色、暖色调、有运动;用户肉眼对比无差别)。
- 产物:`output/video/MiniMax_H3_solattn_10step_00001_.mp4`。
- 未探索:tau=1.5(更稀疏更快)/ morton 3d / Sol-Engine 内核融合(论文宣称的 2x 场景)。

### 20 步对比实测(2026-08-05 19:25)

| 指标 | 基线 20 步(fp8 无加速) | SolAttn 20 步 |
|---|---|---|
| 稳态每步耗时 | ~12.7s | **~10.15s(↓20%)** |
| 前 20%(dense 阶段) | — | 15.5s/步 |
| 总耗时(模型已缓存) | ~6 min(采样 4:15 + 解码) | **266s ≈ 4.4 min(↓~26%)** |
| **VRAM 峰值** | 14.2-14.5GB(87%) | **14.7GB(92%)** |
| **GPU 利用率** | 100%(采样期) | **峰值 100% / 平均 88%** |
| RAM(进程工作集)峰值 | ~25-45GB | **45GB** |
| 系统 RAM 峰值 | ~48GB | **58GB(64GB 的 90%)** |

- 20 步输出质量:正常(std 34-61、720+ 唯一色、暖色调,与基线一致)。
- 说明:SolAttn 收益集中在稀疏阶段(后 80%),前 20% dense 阶段和基线一样;整体省时 ~26%(模型缓存下)。GPU 满载,显存比基线略高(92%)因 SolAttn 需要额外 proxy/scale 张量。
- 产物:`output/video/MiniMax_H3_solattn_20step_00001_.mp4`;监控数据:`comfyui_download/solattn_20step_monitor.csv`。

*文档更新至 2026-08-05 19:30*

---

## 阶段 7:pysssss H3 无缝拼接工作流 + 耗时实测(2026-08-11)

研究了 ComfyUI-wiki 的 `pysssss-workflows`(卡夫卡《变形记》多段无缝拼接,T2V/R2V),完整报告见 `docs/research-pysssss-h3-workflows.md`。

### 环境落地(全部复用现有 python_embeded 和模型)
- 三节点包隔离克隆到 `F:\python\h3\example\`(rgthree-comfy / ComfyUI-H3-Motion-Context / ComfyUI-KJNodes),`extra_model_paths.yaml` 注册为额外 custom_nodes 目录,不动 `custom_nodes\`。
- 补 KJNodes 依赖(color-matcher / matplotlib / mss / opencv-python-headless);sageattention 原本就有。
- 三个 `*.bat` 加 `chcp 65001` + `PYTHONUTF8=1`(修 rgthree 🎉 在 GBK 控制台崩启动);`comfyui-frontend-package` 1.47.11→1.48.7(修前后端版本警告)。
- 所有工作流/模板的模型名从 nvfp4/int8 批量映射到本地 fp8_scaled / int8_convrot。

### 耗时实测(864×480 / 124帧 / 20步,RTX 5060 Ti 16GB)
| 运行 | 总耗时 | 模型初始化 | 采样 | 其余 |
|---|---|---|---|---|
| 段01 全 SolAttn(morton/sink 开) | **11:55** | 3:30 | 2:58(20步) | ~5:27(解码+音频+合成+保存) |
| 段01 共存降级版(morton/sink 关) | 12:55 | 4:19 | ~2:40 | ~5:50 |
| 段01+段02 无缝链条 | 16:14 | 3:51 | 段01+段02 | 两段解码+合成 |

- **采样步速**:dense 阶段 ~16.5s/步(EasyCache 前 20% 未复用),跨步复用后降到 **~3.6-5.4s/步**。
- **结论**:morton/sink 开关对耗时几乎无影响(全 SolAttn 反比降级版快 1 分钟,因 triton autotune 缓存命中);相对纯 fp8 基线采样 4:15,`SolAttn+EasyCache` 采样 2:58,**省 ~30%**。
- 上表均为**冷启动**(改配置重启 ComfyUI 需重载模型 3-4 分钟);热启动单段 ≈ 8.5 分钟(采样~3 分 + 解码~5 分半)。

### SolAttn / H3-Motion-Context 二选一开关
两个包都要独占 `PackedLayout.__init__`,无法共存。开关在 `custom_nodes\ComfyUI-SolAttn_triton\_morton_h3.py` 的 `_SKIP_LAYOUT_PATCH`:
- **True** → SolAttn 不接管布局(注意力内核照常,morton/sink 停)→ H3-Motion-Context 无缝链条可用。
- **False(当前)** → SolAttn 全开(morton + conditioning sink)→ 链条停用。
- 切换后需重启 ComfyUI。已验证两档均输出健康(std 45.4~46.5、~1.7 万唯一色、运动曲线正常)。链条档接缝:clip01 末帧 vs clip02 头帧均差 4.09/255(内部相邻仅 2.09)→ 视觉连续。

*文档更新至 2026-08-11*
