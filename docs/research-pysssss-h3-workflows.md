# 研究：pysssss-workflows 里的 H3 无缝拼接工作流如何应用到本机

**日期**: 2026-08-11
**研究方式**: Perplexity pro（1 主搜索 + 1 follow-up）+ GitHub 代码搜索定位 + 源码直查 + 本地环境实证
**研究对象**: https://github.com/602387193c/ComfyUI-wiki/tree/main/pysssss-workflows 中两条 MiniMax H3 工作流
**目标**: 判断并落地这套工作流到本机（ComfyUI 便携版 v0.30.0@master、RTX 5060 Ti 16GB、CUDA 12.8、MiniMax H3 fp8_scaled）

---

## 关键发现（按置信度排序）

### 1. 这两条 H3 工作流是什么（置信度：高）
- `文生视频_minimax_h3_t2v_Accel_EasyCache_Sage_SolAttn_无缝拼接.json`：T2V 多段无缝拼接测试片（卡夫卡《变形记》开篇，T2VA 文生视频+音频），默认 4 段 ≈58s。
- `参考生视频_minimax_h3_r2v_Accel_EasyCache_Sage_SolAttn_无缝拼接.json`：R2V 参考生视频续篇，默认 4 段（段05 已写好但 bypass），开满 5 段 ≈72s。
- 每段结构：`提示词 → EasyCache → Sage(KJNodes) → SolAttn → 调度器/种子 → Guider + Motion Context → 采样 → 解码画面/声音 → 裁掉重复头 → 合成 → 保存`。
- **加速三件套**（EasyCache + SageAttention + SolAttn）串联在 model 链上（实测连线：`UNETLoader → EasyCache → PathchSageAttentionKJ → SolAttnPatch`）。EasyCache 跨步复用采样缓存（阈值 0.3，起止 0.2~0.9）。
- **无缝拼接原理**：段01 起每段用 `MiniMaxH3MotionContext` 把上一段末尾 22 帧 latent（画面+声音）钉进本段作 context，采样后由 `MiniMaxH3MotionContextTrim` 裁掉重复头，段净增 14.2s；段间通过 `SaveLatent/LoadLatent`（clip_index 槽位，`output/h3_context/...`）跨重启续跑。
- **时长控制**：rgthree `Fast Groups Bypasser` 一键 bypass 尾部段落（只能从尾巴往回关，段01 是起点永远保留）。

### 2. 目录名 "pysssss-workflows" ≠ 需要 pysssss 节点包（置信度：高）
- 该目录是 pysssss（ComfyUI-Custom-Scripts 作者）的个人工作流合集；两条 H3 工作流**不含任何 ComfyUI-Custom-Scripts 节点类型**，无需安装 pysssss 插件包。

### 3. 需要补装的包只有 3 个（置信度：高，全部经 GitHub 源码定位）
| 包 | 提供节点 | 出处文件 |
|---|---|---|
| `rgthree/rgthree-comfy` | `Fast Groups Bypasser (rgthree)`、`MarkdownNote` | `web/comfyui/constants.js`（前端注册节点） |
| `NikoDemon80/ComfyUI-H3-Motion-Context` | `MiniMaxH3MotionContext` / `Trim` / `SaveLatent` / `LoadLatent` | `nodes.py`（v0.2.0，2026-08-09） |
| `kijai/ComfyUI-KJNodes` | `PathchSageAttentionKJ` | `nodes/model_optimization_nodes.py` |

### 4. 环境其余部分本机已全部就绪（置信度：高，本地实证）
- **EasyCache**：本机 ComfyUI 原生已有 `comfy_extras/nodes_easycache.py`（class EasyCache，reuse_threshold/start_percent/end_percent/verbose 齐全）✓
- **SolAttnPatch**：已装 `custom_nodes/ComfyUI-SolAttn_triton` ✓
- **sageattention**：python_embeded 已装 `sageattention 2.2.0+cu130torch2.10.0andhigher.post6`，**冒烟测试通过**（`sageattn` 与 `sageattn_qk_int8_pv_fp16_triton` 在 sm_120 上均跑通）✓
- **triton-windows 3.7.1.post27** + CPython 头 + `python313.lib`（SolAttn 修复痕迹，memory 有记录）✓
- **原生 H3 节点**：`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` / `MiniMaxH3SigmaShift` 已在 `comfy_extras/nodes_minimax_h3.py` ✓

### 5. 模型路径必须重映射（置信度：高，已核对工作流 JSON 与本地 models 目录）
| 工作流要求 | 本机已有（已验证可用） |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`（fl2va） |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`（ref2va，R2V 用） |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（text encoder） |
| `minimax_h3_video_vae_fp16.safetensors` / `minimax_h3_audio_vae_fp32.safetensors` | 同名 ✓ |

> 无需下载任何新权重（本地 fp8_scaled + int8 te 就是你已验证可用的配置）。只需在 ComfyUI 里改 3 个加载器文件名。CLIPLoader type 两边都是 `minimax`，不用改。

### 6. ComfyUI 内核兼容风险（置信度：中高）
- H3-Motion-Context v0.2.0（2026-08-09 提交）采用**运行期 patch + 自检**：首次运行才装 layout/payload patch，每次启动对照 live ComfyUI 代码校验，假设被打破就**拒绝运行并打印原因**（README 明确承诺"loud failure beats a bad render"）。
- 本机 ComfyUI vendored 于 2026-08-08（master），与该包同时代，兼容概率高；但**是否完全兼容须装完跑一次才知道**（自检会明说）。
- 该包要求 context_length 必须是 5/22/39/56（整数 latent step），当前工作流用 22；生成帧长按 VIDEO_RUN_GRID (124,107,90,73,56,39,22,5,1) 向下吸附。

### 7. 16GB 显存压力与首跑建议（置信度：中）
- 工作流默认 **1376×768、每段 362 帧 @24fps、20 步、4~5 段**——远超你验证过的 864×480/124 帧，直接全量跑大概率 OOM/超时。
- 建议首跑只开段01（rgthree 控制台 bypass 02~04，或 Ctrl+B），分辨率降到 864×480、length 改到 124 附近，先验证 Motion Context patch 安装成功、段01 输出正常，再逐步加段。

---

## 项目文档审查

- `F:\python\h3\docs.md`（RELIABLE，本会话亲历的操作记录）：fp8_scaled 权重已跑通、SolAttn 20 步加速记录、显存 14.7GB/16GB 峰值 —— 作为"本机基线"与"模型路径映射"的依据。
- `F:\python\h3\research-h3-garbage.md`（RELIABLE）：H3 乱码根因（下载损坏）与 Perplexity 占位引用问题 —— 本次研究也复现了后者。
- `F:\python\h3\README.md`（RELIABLE，简述性）。
- 本地 ComfyUI 源码检查：`comfy_extras/nodes_easycache.py`（EasyCache 原生）、`comfy_extras/nodes_minimax_h3.py`（无 MotionContext 类）、`custom_nodes/`（仅 SolAttn）。
- 本地 python_embeded 包检查 + sageattention 冒烟测试（RELIABLE，实测）。
- 两条工作流 JSON 已下载到 `comfyui_download/pysssss_h3/`（本次新取，含节点类型/连线/模型名/内嵌说明文档）。

---

## 来源（可验证）

- `602387193c/ComfyUI-wiki`：`pysssss-workflows/` 目录，两条 H3 工作流原始 JSON。
- `NikoDemon80/ComfyUI-H3-Motion-Context`（`nodes.py`、README、v0.2.0 commit 2026-08-09）：MotionContext 节点定义与兼容自检机制。
- `kijai/ComfyUI-KJNodes`：`nodes/model_optimization_nodes.py` 定义 `PathchSageAttentionKJ`（sageattn_modes 含 cuda/triton 变体，ImportError 优雅降级）。
- `rgthree/rgthree-comfy`：`web/comfyui/constants.js` 前端注册 `Fast Groups Bypasser (rgthree)` / `MarkdownNote`。
- `Comfy-Org/ComfyUI`：`comfy_extras/nodes_easycache.py`、`comfy_extras/nodes_minimax_h3.py`。
- Perplexity pro：主搜索（backend 9a169411…）+ follow-up（backend b611e6c6…）—— 具体结论未直接采用（见矛盾与缺口）。

---

## 矛盾与缺口

- **Perplexity 结论不可靠**：两次回答均为泛泛断言、无实质 URL，把 EasyCache 误判到 rgthree、把 MotionContext 误判到 KJNodes/官方模板。与你既往"占位 [1][2]"体验一致。本报告全部改用 **GitHub 代码搜索 + 源码直查**定位，结论可逐行复核。
- **未实机验证**：H3-Motion-Context 与本机 vendored ComfyUI 的运行期兼容性、Sage+SolAttn 串联的实测效果与峰值显存，均需安装后实跑确认。
- **Sage 与 SolAttn 串联语义**：两者都是 attention patch，SolAttnPatch 在链上靠后，理论上其 patch 会覆盖 Sage 的 patch。pysssss 保留两者更可能是"二选一/便于 bypass"而非真正叠加；建议首跑保留 Sage（环境已就绪），若异常再 bypass 其一。

---

## 建议（可操作下一步）

1. **装 3 个包**（git clone 到 `ComfyUI_windows_portable/ComfyUI/custom_nodes/`）：
   ```bash
   git clone https://github.com/rgthree/rgthree-comfy.git
   git clone https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git
   git clone https://github.com/kijai/ComfyUI-KJNodes.git
   ```
2. **重启 ComfyUI**，加载 `comfyui_download/pysssss_h3/文生视频_..._无缝拼接.json`。
3. **改 3 个加载器文件名**（模型映射表见发现 5），其余不动。
4. **首跑瘦身**：rgthree 控制台只留段01；分辨率降到 864×480、length 改到 124；确认 `h3_motion_context: nodes registered / interior keyframe anchors enabled` 日志出现、输出非乱码，再逐步加段/升分辨率。
5. **R2V 前准备参考图**：把工作流引用的 `ComfyUI/input/h3_char_ref.png` 换成你自己的参考图（所有段共用左上角 LoadImage）。
6. 若 Motion Context 拒绝运行：按日志提示核对 ComfyUI 内核版本（可 `git pull` 更新 vendored ComfyUI 到最新 master 再试）。

---

## 搜索覆盖

- **本地**：项目 docs 扫描；ComfyUI 源码节点检查（EasyCache/MiniMaxH3/custom_nodes）；python_embeded 包清单 + sageattention 冒烟测试；两条工作流 JSON 全量解析（节点类型、连线、模型名、内嵌使用说明）。
- **GitHub**：代码搜索定位 `MiniMaxH3MotionContext`/`EasyCache`/`PathchSageAttentionKJ`/`MarkdownNote` 定义仓库；仓库 README、源码、提交历史直查。
- **Perplexity**：1 次主搜索（pro，6 项清单）+ 1 次 follow-up（聚焦节点归属）。
- **未覆盖**：实机运行验证（待装包后）；上游 ComfyUI issue 逐条核实；sageattention 与 SolAttn 在同工作流中的实测对比。

---

## 落地状态（2026-08-11 实操）

已完成的环境改造（全部保留可逆）：

1. **隔离克隆**：三个仓库已克隆到 `F:\python\h3\example\`（rgthree-comfy / ComfyUI-H3-Motion-Context / ComfyUI-KJNodes），各自独立子目录，`/example/` 已加入 `.gitignore`。
2. **加载方式**：新建 `ComfyUI\extra_model_paths.yaml`，把 `custom_nodes: ../../example` 注册为额外自定义节点目录 —— ComfyUI 直接扫描 example/ 下的三个仓库，源码零复制、不动 `custom_nodes\`。
3. **依赖补齐**（python_embeded 复用）：`color-matcher`、`matplotlib`、`mss`、`opencv-python-headless` 已安装（KJNodes 要求）。sageattention 2.2.0+cu130 / SolAttn 原本就有，冒烟测试通过。
4. **启动修复**：rgthree 打印 🎉 横幅在 GBK 控制台抛 `UnicodeEncodeError` 导致启动崩溃 → 三个 `*.bat` 已加 `@echo off` + `chcp 65001 >nul` + `set PYTHONUTF8=1`。
5. **前后端版本**：`comfyui-frontend-package` 1.47.11 → **1.48.7**（后端要求 ≥1.47.12）。
6. **启动验证通过**：`PYTHONUTF8=1` 下 ComfyUI 0.30.0 正常启动，`rgthree-comfy` / `h3_motion_context: nodes registered` / `ComfyUI-KJNodes` 全部注册，服务器就绪；API 确认 `MiniMaxH3MotionContext*` / `PathchSageAttentionKJ` / `EasyCache` / `SolAttnPatch` / `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` 均已注册（`Fast Groups Bypasser (rgthree)`、`MarkdownNote` 为前端节点，不在 object_info，属正常）。
7. **首跑瘦身版**：`comfyui_download\pysssss_h3\h3_t2v_seamless_slim01_fp8.json` —— 模型名重映射到本地 fp8_scaled / int8_convrot，段01 降为 864×480/124 帧，段 02-04（45 节点）已 bypass。

**待办**：加载瘦身版跑通段01（验证 Motion Context patch 安装、输出非乱码）；R2V 需先放好参考图；之后按需在 UI 里升分辨率/加段。

---

## 端到端验证（2026-08-11 实测通过 ✅）

用 API 方式（`wf2api.py` 把 UI 工作流转 API prompt，POST `/prompt`）跑了瘦身版段01：

- **产物**：`ComfyUI\output\video\Kafka_T2V\clip01_00001_.mp4`（864×480@24fps，124 帧，含 AAC 音频）；续跑 latent `output\h3_context\Kafka_T2V\clip_00001.safetensors`。
- **加速链生效**：SolAttn 内核运行（`sparse ... tau=1.2 int8 pointer`），每步从 16s 降到 3.5s（EasyCache 跨步复用 + SolAttn 稀疏阶段）。Sage（PathchSageAttentionKJ）在链上但 SolAttn 靠后实际生效（符合"SolAttn patch 覆盖 Sage"的判断）。
- **输出健康（非乱码）**：帧 std 45.4~45.7、16968 唯一色、运动曲线随间隔递增（0.04s→1.67，0.42s→9.12，1.25s→14.50，慢推镜头）；mp4 含 video+audio 两轨；H3-Motion-Context 日志确认 A/V 同步 drift 0.01ms。
- **总耗时 12:55**（含 ~4:19 模型加载 + JIT；纯采样 ~2:40）。
- 顺带修复：API 转换需把链接节点 id 用字符串（`["110",0]`），否则 `execution.py` KeyError；SaveVideo 的 `codec`（COMFY_DYNAMICCOMBO_V3）需识别为 widget。
- **模型名修复**：两个原版工作流（T2V/R2V）的 `widgets_values` 已从 nvfp4/int8 映射到本地 fp8_scaled / int8_convrot（与原报告发现 5 一致）。全部 14 个工作流/模板 JSON 均批量映射。

---

## 段01+段02 无缝链条实测（2026-08-11 通过 ✅）

- **冲突与修复**：SolAttn_triton 的 `_morton_h3` 无条件接管 `PackedLayout.__init__`，H3-Motion-Context 检测到 foreign wrapper 拒绝运行（"Disable one of them and restart"）。**修复**：在 `custom_nodes\ComfyUI-SolAttn_triton\_morton_h3.py` 加 `_SKIP_LAYOUT_PATCH=True`，让 SolAttn 不独占布局补丁（注意力内核独立于布局，仍运行；morton 重排 + conditioning sink 优雅降级）。
- **运行结果**：`h3_t2v_seamless_s0102_fp8.json`（段01+02，均 864×480），`Prompt executed in 00:16:14`，`status: success`。
  - `clip01_00002_.mp4`（124 帧）+ `clip02_00001_.mp4`（**102 帧** = 124−22 帧 motion-context 重叠，`tail trimmed 267 samples`）。
  - H3-MC 日志：`interior keyframe anchors enabled` / `keyframe/ref coexistence enabled` / 两段 latent 已存 `h3_context\Kafka_T2V\clip_0000X.safetensors`。
- **接缝验证**：clip01 末帧 vs clip02 头帧均差 **4.09/255**（clip02 内部相邻帧仅 2.09）→ 同一场景视觉连续；随距离递增（5帧→8.04，20帧→15.20）→ 运动延续。两段像素均健康（std 44.6~49.5，~1.8 万唯一色）。
- **结论**：EasyCache + SageAttention + SolAttn（共存降级）+ MotionContext 无缝拼接全链路在本机 16GB 显存可用。
- **后续反转（2026-08-11）**：用户选择**恢复 SolAttn 完整功能**，`_SKIP_LAYOUT_PATCH` 翻回 **False**。SolAttn morton + conditioning sink 全部恢复（实测 `conditioning sink: KV blocks exact` + sparse int8 内核 + 无 "no layout registered"），段01 输出验证健康（std 45.4~46.5、17106 唯一色、运动正常）。**代价**：H3-Motion-Context 会因布局补丁被 SolAttn 接管而拒绝运行 → **多段无缝拼接链条当前不可用**（二选一）。若需恢复链条，把 `_SKIP_LAYOUT_PATCH` 翻回 True 并重启。
