# 研究:SageAttention2/3 能否加速 MiniMax-H3(ComfyUI 本地)

**日期**: 2026-08-05
**提问**: 用 SageAttention2(sage2,内部把 Q/K 量化为 int8)加速 ComfyUI 的 MiniMax-H3 视频生成?
**环境**: Windows 11 · RTX 5060 Ti 16GB(Blackwell, sm_120, compute 12.0)· CUDA 13.0 · ComfyUI 0.30.0(master)· torch 2.13.0+cu130 · comfy-kitchen 0.2.26 · fl2va fp8_scaled + qwen3vl int8_convrot 文本编码器 · 当前 480p/864×480/124帧 ≈ 12-16s/步。

---

## 核心结论(一句话)

**SageAttention2 对 H3 的主采样热路径必然无效——ComfyUI 的 H3 实现显式传了 `low_precision_attention=False`,`attention_sage` 首行判断就回退到 PyTorch SDPA。** 想用只能试 SageAttention3,但它是绕过 H3 显式高精度要求的赌博,且对"注意力只占一部分"的 33B 扩散管线收益有限、风险不低。

---

## Key Findings(按置信度)

### [高] H3 主 transformer 的 attention 走标准分发器,但显式禁止低精度注意力
本地源码(ComfyUI 0.30.0 `comfy/ldm/minimax/model.py`):
- `model.py:27` 导入 `optimized_attention`(标准分发器)→ H3 与 ComfyUI 内置 attention 切换完全兼容。
- `model.py:181-182` 调用:`optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, low_precision_attention=False, ...)`。
- ComfyUI `comfy/ldm/modules/attention.py:549-550` `attention_sage` 首行:`if kwargs.get("low_precision_attention", True) is False ...: return attention_pytorch(...)`。
- 结论:即使 `--use-sage-attention` 生效,H3 每个 transformer block 的 attention 都会**立即回退到 `attention_pytorch`(SDPA)**,sage2 的 int8 Q/K 内核不会运行。这是硬逻辑,不是性能问题。

### [高] ComfyUI 0.30 内置 SageAttention,但缺包会直接 exit(-1)
- `attention.py:20-35`:如果 `--use-sage-attention` 且 `sageattention` 未安装 → 打印错误并 `exit(-1)`(ComfyUI 直接启动失败)。
- 当前 portable python **没有 triton、没有 sageattention**(实测 `ModuleNotFoundError`)。
- 启示:不能随手加 `--use-sage-attention`,必须先装好 wheel 且装错会导致启动失败。

### [中高] H3 的 VAE/text-encoder attention 默认可用 sage,但不在关键路径
- `comfy/ldm/minimax/vae.py:239` 的 VAE attention 未传 `low_precision_attention`,默认 True → `--use-sage-attention` 时**能**被 sage 命中。
- 但 VAE 解码只占一次生成的 ~1-2 分钟(共 ~15 分钟);文本编码器只跑一次。**对 20 步采样主循环贡献≈0**。

### [中] SageAttention3 是唯一理论上能碰到 H3 attention 的路径,但有硬门槛
- `attention3_sage`(`attention.py:596`)只检查 `q.dtype not in (fp16, bf16)`、形状、`dim_head >= 256`、`N <= 1024` 才回退;H3 的 q 是 bf16、head_dim 通常 128、N(480p 长 latent)>1024 → **sageattn3_blackwell 理论上会被调用**。
- 但它**绕过了 H3 显式请求的高精度 attention**(`low_precision_attention=False` 被无视)→ 输出质量有回退风险,可能重新踩上"乱码/质量劣化"的老坑。
- 安装面:需 `sageattn3` + 匹配 cu130/sm_120 的 Windows wheel(见 Sources),要装进**正在工作的** portable python,有破坏现有环境的风险。

### [中] 注意力可能不是 H3 采样主瓶颈(工程估算,未 profile)
- 33B transformer 每步主要是**权重矩阵乘**(fp8 tensor core 路径),注意力是其中一部分;RTX 5060 Ti 16GB 只有 ~1.8GB VRAM 余量,层还要在 RAM↔VRAM 间流动。
- 但实测采样时 GPU 利用率 100%,倾向 compute-bound。没有 ncu/profile 前,**attn 占比无法量化**;sage 宣称的"2-3x"来自 attention-bound 的长上下文 LLM 场景,不能直接套到 33B 扩散管线。
- 因此即便 sage3 生效,整体提速预计有限(<20%),且以质量为赌注。

---

## 项目文档审查

| 文档 | 状态 | 相关结论 |
|---|---|---|
| `F:\python\h3\docs.md`(阶段 5) | RELIABLE | 干净权重 + fp8_scaled 已一次通过;~15.5min/生成;fp8 ops 均走 comfy-kitchen |
| `F:\python\h3\research-h3-garbage.md` | RELIABLE(历史) | 乱码根因是下载损坏;ComfyUI H3 支持早期 |
| 本机运行日志 `comfyui_run3.log` | RELIABLE | `model weight dtype bfloat16`、comfy-kitchen cuda backend 可用、triton 不可用、峰值 VRAM 14.2GB |
| 本机 ComfyUI 源码 `comfy/ldm/minimax/*`、`attention.py` | RELIABLE(权威,当前版本) | 上述 [高]/[中] 结论的直接证据 |

---

## Sources(2026-08-05 经 Perplexity auto 检索,部分 GitHub 直连被网络策略拦截)

- SageAttention2 Windows wheel(2.2.0,sm_120/RTX 50): https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows
- SageAttention 2.2.0 Windows/5090 构建说明: https://huggingface.co/nhathoangfoto/SageAttention2
- SageAttention-for-windows 分支: https://github.com/sdbds/SageAttention-for-windows
- ComfyUI-SageAttention3 扩展(comfy.icu): https://comfy.icu/extension/wallen0322__ComfyUI-SageAttention3
- ComfyUI-SageAttention3 安装指南(runcomfy): https://www.runcomfy.com/comfyui-nodes/ComfyUI-SageAttention3
- ComfyUI RTX 50 系/Windows attention 讨论: https://github.com/Comfy-Org/ComfyUI/discussions/11583
- triton+sageattention 一键安装器(Windows/5090): https://github.com/DazzleML/comfyui-triton-and-sageattention-installer/releases
- SageAttentionV3 对 Blackwell 其他 GPU 的支持 issue: https://github.com/thu-ml/SageAttention/issues/237
- sageattn3 cu130/5090 端点 README: https://huggingface.co/Seryoger/Sageattention-3-cu130-5090-endpoint

> 注:GitHub/网页直连在本网络被拦截,上述 URL 来自 Perplexity 检索结果,未逐一在线复核;结论以本地 ComfyUI 源码证据为准。

---

## 矛盾与缺口

- **sage3 质量风险未实测**:`low_precision_attention=False` 是 H3 移植者的显式选择,但未在 H3 上对比过 sage3 vs SDPA 的生成质量。
- **attn 占比未 profile**:没有 ncu/nsys 数据,无法确定注意力在主循环里的真实权重。
- **sage3 的 cu130/sm_120 Windows wheel 是否真可装、装后性能如何**,本报告未验证(需先装上才能测)。
- Perplexity pro 模式后端降级(返回"无实时联网"),研究主要依赖 auto 模式 + 本地源码。

---

## 建议(行动项)

1. **不要为 H3 装 SageAttention2**:代码层面已证明对采样主循环无效(必然回退 SDPA)。装它只有风险没有收益。
2. **想试就只试 SageAttention3**,且先备份当前 portable python(`python_embeded` 可整体复制)或记录可还原点;装完先用 `--use-sage-attention` 单独验证加载,再对比生成帧质量(别只看速度)。
3. **更划算的提速杠杆**(风险低、收益明确):
   - 出片预览用 **10 步**(docs 记录 ≈ 2 分钟)或更小分辨率(如 720×400)→ 最大杠杆。
   - 帧数 124→62(约 2.6s)做快速迭代。
   - 试 ComfyUI `--fast`(torch.compile)观察是否减少每步开销(注意 H3 有 3:48 的 JIT 初始化,编译可能加速后续步)。
4. 若要 profile 到底哪里慢,再决定是否上 sage3:nvidia-smi 测步间显存占用/利用率 + 一段 `torch.profiler`。

---

## 补充调查(SDPA / row-wise,2026-08-05)

### SDPA:已经在用,且已是最优,无可再榨
- H3 的 attention 现在就跑 SDPA(run3 日志 "Using pytorch attention";`attention_pytorch` → `comfy.ops.scaled_dot_product_attention`)。
- `comfy/ops.py:56-90`:CUDA 上带 **后端优先级 FLASH → CUDNN → EFFICIENT → MATH**(`set_priority=True`),且 `model_management.py:524-526` 全开后端。
- 实测(sm_120, bf16, (1,32,8192,128), 无 mask):auto / flash / cudnn / efficient **全部 ≈ 58ms**,无更优内核。sage 回退到的就是它,不用再动。

### row-wise(int8 行级)量化:Blackwell 上无提速空间
- 当前 fl2va = fp8 逐张量,text encoder = int8_tensorwise+convrot;**两者都已走融合 tensor-core matmul**(`ops.py:856` `fp8_linear`→TensorCoreFP8Layout 融合 fp8;`ops.py:992` `int8_linear`→融合 int8,带 convrot),无"反量化到 bf16"的浪费。
- sm_120 上 **INT8 与 FP8(E4M3)峰值吞吐相同(均 2× FP16)**。row-wise 只是把缩放粒度改到"每行"(精度特性变化),**不改吞吐**。
- 换 fl2va 到 int8_rowwise:int8_convrot 文件(~19.5GB,此前下载损坏的那个)需重下/重验 + 质量变动风险,**换不来速度**。不推荐。

---

## 实证测试:SageAttention 对 H3(2026-08-05 17:06)

按 wildminder 指南实测(安装 triton-windows 3.7.1 + sageattention 2.2.0 cu130/abi3 wheel,ComfyUI 加 `--use-sage-attention`),用 torch.profiler 单步对比:

| 内核 | 无 sage(基线) | 开 sage |
|---|---|---|
| attention(cuDNN flash f16) | 7702ms | **7679ms(不变)** |
| fp8 矩阵乘 | 5112ms | 5100ms(不变) |
| bf16 矩阵乘 | 2520ms | 2510ms(不变) |
| sage 相关内核 | 0 | **0** |

**结论:零 sage 内核被调用。** H3 主 transformer 52 次 attention 全部仍走 cuDNN SDPA。`low_precision_attention=False`(`minimax/model.py:182`)使 `attention_sage`(`attention.py:549`)首行**静默**回退 pytorch,连报错都没有。sageattn 本身在 sm_120 可运行(实测与 SDPA max diff 0.006),是 ComfyUI 的 H3 实现主动绕开它。指南"接近翻倍"对 SDXL/Flux/Wan 成立,对 H3 不成立。

> 环境已还原为无 flag 启动;triton-windows + sageattention 仍装在 portable python(无 flag 时惰性,不干扰);回滚清单:`comfyui_download/pip_before_sage.txt`。

---

## RTX 5060 Ti 的 FP8 实测(2026-08-05)

**问题:5060 Ti(Blackwell sm_120)有没有 fp8 优化?**

**硬件有 fp8 tensor core,但当前软件栈给不了原生 fp8×fp8 矩阵乘。**

实测(torch 2.13.0+cu130 / cuBLAS 13 / sm_120):
- `torch._scaled_mm`(fp8×fp8):**所有尺寸(512~6144)一律 `CUBLAS_STATUS_NOT_SUPPORTED`**
- `torch.nn.functional.scaled_mm`(新版 API + TensorWise):**同样 NOT_SUPPORTED**
- 但 H3 的 fp8 矩阵乘**正常跑了**(profile 里 150 次,5.1s),内核名 `sm89_xmma_gemm_e4m3bf16_e4m3f32`——是 **fp8(激活)×bf16(权重)混合 GEMM + sm89(Ada)时代内核**,经 PTX 前向兼容在 sm_120 上跑。
- run 日志 0 条 "Exception during fp8 op" → ComfyUI 走的是混合内核路径,没有尝试会失败的 fp8×fp8。

**结论**:fl2va 的 fp8_scaled 在 5060 Ti 上**没有吃到原生 Blackwell fp8**(cuBLAS 13 对 sm_120 不提供 fp8×fp8 算法),实际以"激活量化 fp8 × 权重 bf16"的混合 GEMM 运行,近似 bf16 吞吐。这是"fp8 化了但没明显提速"的软件层原因。

**未决问题**(Perplexity 后端降级,未能核实):更新版 cuBLAS(>13)或特定 torch build 是否已为 sm_120 提供原生 fp8×fp8;官方 5060 Ti fp8 TFLOPS 指标。

---

## Sol-Attn 实测成功(2026-08-05,更新)

`kijai/ComfyUI-SolAttn_triton`(Sol-Attn,arXiv 2607.24027)专为 MiniMax H3 做的**免训练稀疏注意力**,最终**成功加速 H3 attention**:

- 卡点:triton 编译 `cuda_utils` 缺 embedded python 的 CPython 头文件 + `python313.lib`。
- 修复:铺入 CPython 3.13 Include 头文件 + 用 **MSVC** 从 `python313.dll` 生成 `python313.lib`。
- 结果:attention 7.7s/步→~4s/步(profile,↓55%),墙钟 ~12.7s→~10.4s/步(↓18%),10 步生成 210s;**质量视觉无损**(用户肉眼对比帧无差别)。
- 详细记录见 `docs.md` 阶段 6。

**教训**:sage(依赖 `low_precision_attention` 开关)对 H3 无效,但 Sol-Attn 通过 `optimized_attention_override` + H3 专属 Morton hook 绕开了这个开关——**针对具体模型架构的注意力加速才是正解**。论文宣称 2x 需配 Sol-Engine 内核融合(未测)。

---

## Search Coverage

- Perplexity(auto 模式):主检索 SageAttention2 sm_120 支持 → 拿到 2.2.0 Windows wheel、sage3、安装器、讨论链接;follow-up 补齐 ComfyUI-SageAttention 集成与 Windows 路径。
- 本地验证(权威):ComfyUI 源码逐行确认 H3 attention 分发、sage 回退逻辑、VAE 路径、dtype;实测 portable python 无 triton/sageattention;GPU/日志事实核对。
- 未覆盖:GitHub 页面直读(被网络策略拦);sage3 实测安装与基准(需先装);H3 attention 占比的 profiler 数据。
- 失败项:Perplexity pro 模式后端降级;WebFetch github.com 被拦。

---

## 实测解锁 SageAttention2:H3 上加速 ~68% 但硬崩 GPU(2026-08-13)

按用户指示实测"拆掉 `low_precision_attention=False` 门后 sage2 能否加速 H3":

**改动**:`minimax/model.py:181-182` 删除 `low_precision_attention=False`;启动加 `--use-sage-attention`(新建 `run_sage_test.bat`)。

**结果(1 步 profile,torch.profiler,H3SampleProfiler)**:
| 内核 | baseline(无 sage) | sage2 解锁 |
|---|---|---|
| attention | cuDNN flash ~7702ms | **sageattn `qk_int8_sv_f8_accum` 2474ms(↓68%)** |

- 52 次 attention 全部走 `sageattention_qattn_sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`(sage 的 int8 QK + fp8 V 内核),不再是 SDPA。**拆门确实解锁了 sage2。**

**结果(20 步墙钟)**:**第 3 步采样时进程 `Fatal Python error: Aborted`**。崩溃栈顶在 `attention.py:592`(`attention_sage` 里 `sageattn()` 之后的输出 reshape)。`attention_sage` 的 `try/except Exception` 兜不住 native abort。随后 **CUDA 驱动被带崩**:`nvidia-smi` → `GPU is lost. Reboot the system to recover this GPU`。

**结论**:sage2 在 H3 上**能加速 attention(~68%)但极不稳定**,多步生成必然硬崩 GPU(需重启系统恢复),比"质量回退"更糟——是稳定性灾难。**H3 作者用 `low_precision_attention=False` 锁死 sage 是有充分理由的**(不只是精度,而是防崩溃)。

### 崩溃根因(2026-08-13 静态代码证据,置信度高)

**一句话**:sageattn 2.2.0 **没有 sm120 专属模块**,在 RTX 5060 Ti 上把 H3 attention **强制塞进 sm89 编译的 fp8-V 融合内核**,该内核经 PTX 前向兼容在 sm120 上跑,多步后触发非法内存访问→原生 abort→驱动重置 GPU。

证据链(全部来自 `python_embeded/Lib/site-packages/sageattention/`):

1. **sm120 分发强制走 fp8-V 路径**(`core.py:171-178`):`sageattn()` 对 sm120 无条件调 `sageattn_qk_int8_pv_fp8_cuda`,`qk_quant_gran="per_warp"`,`pv_accum_dtype="fp32+fp16"`(CUDA 13≥12.8 → SageAttention2++)。注释原文:**"triton kernel is currently not usable on sm120"**——sm120 连 fp16-triton 回退都没有。
2. **fp8_cuda 调的是 sm89 编译模块**(`core.py:767-773`):实际跑的内核 = `_qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`(profile 里 `sageattention_qattn_sm89::...` 的 2474ms 就是它)。包内只有 `_qattn_sm80/_sm89/_sm90.pyd`,**没有 sm100/sm120 模块**。
3. **所以 sm120 上这是"sm89 内核经 PTX JIT 前向兼容到 Blackwell 消费卡"**——作者根本没为 sm120 编译/验证这条路径(有 sm120 支持就不会没有 sm120 模块,也不会注释掉 triton 回退)。
4. **与已知 sm120 fp8 脆弱性吻合**(本报告上部实测):cuBLAS 13 对 sm120 原生 fp8×fp8 = `CUBLAS_STATUS_NOT_SUPPORTED`,本项目自己的 fp8 也是靠 sm89 前向兼容混合 GEMM 跑的。fp8 tensor-core 在这张卡/这套软件栈上本就不是稳的。
5. **崩溃机制**:sm89→sm120 的 JIT 融合内核在多步负载下越界写/不可恢复硬件错误(异步)→ `attention_sage` 的 `except Exception` 抓不到 → 在下一个同步点(`attention.py:592` reshape)才浮出 → torch fatal handler `Abort` → 驱动升级为 GPU reset。**"第 3 步崩"呈内存安全问题特征**(前几步能跑、跑一会儿才爆),不是确定性的形状/逻辑错误。

**验证方向(未执行,需再冒险 GPU)**:用 fp16-V 变体 `sageattn_qk_int8_pv_fp16_cuda`(sm80 用的那条,**不含 fp8 V**)替代 fp8-V 跑 H3——若稳定,则铁证 fp8-V 是元凶;若同样崩,则 sm89 的 int8-QK 内核在 sm120 本身就有问题。这条 fp16-V 路径正是 KJNodes `PathchSageAttentionKJ` 的 `sageattn_qk_int8_pv_fp16_cuda` 模式(内置 `attention_sage` 选不了,只能走节点)。

**后续**:sol+attn2(SolAttn 委托稠密 attention 给 sage2)未测——GPU 已崩需重启,且 sage2 单跑就崩,组合大概率同样崩。环境已还原(`git checkout` model.py + 删 `run_sage_test.bat`)。
