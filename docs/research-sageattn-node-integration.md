# 研究：SageAttention2 为何在 ComfyUI（H3）里不生效，以及如何添加 attention 节点

**日期**: 2026-08-13（2026-08-24 更新后置节点与日志判据）
**研究方式**: Perplexity pro（网络结果未提供可核验源码引用）+ 本地 ComfyUI/KJNodes/SolAttn 源码逐行核对 + 运行日志与 torch.profiler 交叉验证
**问题**: 为什么 SageAttention2 在本机 ComfyUI 里没用上？后置添加 Sage Attention KJ 节点并看到 `Using sage attention` 是否代表已经生效？

---

## 关键发现（按置信度排序）

### [高] 核心根因：`low_precision_attention=False` 让 sage 在所有入口都被静默回退
H3 的 attention 显式要求高精度注意力，而这个开关是 SageAttention 的**硬门**，无论走哪条路都会先判它：

- H3 调用点 `comfy/ldm/minimax/model.py:181-182`：
  ```python
  out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True,
                            low_precision_attention=False, transformer_options=transformer_options)
  ```
- ComfyUI 内置 sage 分发 `comfy/ldm/modules/attention.py:549-550`：
  ```python
  def attention_sage(q, k, v, ...):
      if kwargs.get("low_precision_attention", True) is False or (...):
          return attention_pytorch(...)   # 立即回退 SDPA
  ```
- **节点方式的 sage 补丁也同样被这个门拦住**（见 [高] 第二条）。

结论：`--use-sage-attention` 即使生效，H3 每个 transformer block 的 attention 都**立即回退到 PyTorch SDPA**，sage 的 int8 Q/K 内核一个都不跑。这是硬逻辑，不是性能问题。

### [高] KJNodes 的 `PathchSageAttentionKJ` 节点也带着同一个门，对 H3 同样无效
`example/ComfyUI-KJNodes/nodes/model_optimization_nodes.py`：
- `get_sage_func()` 返回的 `attention_sage` 首行（第 62-63 行）：
  ```python
  if kwargs.get("low_precision_attention", True) is False:
      return attention_pytorch(...)
  ```
- 补丁方式（第 134 行）：
  ```python
  model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
  ```
  —— 与 SolAttn 用的是**同一个 `optimized_attention_override` 通道**，但 SolAttn 的 hook 不判 `low_precision_attention`，所以 SolAttn 能命中 H3，而 Sage 不能。

**结论：即使把 `PathchSageAttentionKJ` 节点接进 H3 工作流（pysssss 工作流就是这么做的），它也会因为 `low_precision_attention=False` 静默退回 SDPA，实际不产生任何 sage 内核。** 项目实测也印证了这一点（`research-pysssss-h3-workflows.md:84` 判断"SolAttn patch 覆盖 Sage"；实测日志显示只有 SolAttn 内核在跑）。

### [高] `Using sage attention mode: auto` 只证明节点完成配置，不证明 Sage 内核执行
KJNodes 在 `get_sage_func()` 一进入时就打印该日志（`model_optimization_nodes.py:28-29`），随后才定义 `sage_func`；真正调用 `sage_func(q, k, v, ...)` 在第 89 行。节点在第 127 行克隆模型，在第 134 行把 wrapper 写入 `optimized_attention_override`。

因此这条日志只能证明：

1. `PathchSageAttentionKJ` 节点运行了；
2. 选择的 Sage 模式可导入；
3. 节点输出的那份 MODEL 当时装上了 attention override。

它**不能**证明 sampler 最终使用了这份 MODEL、override 未被下游节点替换、H3 没触发 fallback，更不能证明 Sage CUDA kernel 已执行。

本项目已有直接反例：`comfyui_download/run_s0102.log:72` 打印 `Using sage attention mode: auto`，第 73 行还显示 SolAttn 成功链上这个 override；但 H3 仍会把 `low_precision_attention=False` 传给 KJ wrapper，并在真正调用 Sage 前回退 SDPA。该日志中可观察到的实际运行证据是第 87/99 行的 SolAttn sparse kernel，而不是 Sage kernel。

### [高] 四层证据必须分开判断
| 层级 | 能证明什么 | 当前可靠判据 |
|---|---|---|
| 1. 节点执行/路由配置 | KJ 节点构造并安装了 override | `Using sage attention mode: ...`；检查最终 MODEL 的 `optimized_attention_override` |
| 2. Sage Python 函数被走到 | wrapper 没在入口 gate 提前回退 | 在第 89 行 `sage_func(...)` 处加运行期计数/日志；当前初始化日志不能证明 |
| 3. Sage CUDA kernel 实际执行 | GPU 确实运行 SageAttention | `torch.profiler` / Nsight 出现 `sageattention_qattn_*`、`sageattention_fused::*`，并有合理调用次数与 CUDA time |
| 4. 稳定可用 | 多步完整生成不崩、质量可接受 | 完整采样、重复运行、质量对比；一次 profiler 命中不等于稳定 |

本项目正例是临时拆除 H3 gate 后的 `profile_summary_attn2.json:31-35`：`sageattention_qattn_sm89::qk_int8_sv_f8_accum...` 共调用 52 次，才足以证明 H3 主 attention 实际执行了 Sage。反过来，该配置随后在 20 步采样第 3 步 native abort（`sage_test.log:161-166`），说明“执行过”仍不等于“稳定可用”。

### [高] 后置节点位置不能绕过 H3 gate，最终 MODEL 连线与后续覆盖仍然重要
`wrap_attn` 在运行时读取 `transformer_options["optimized_attention_override"]` 并调用当前值（`attention.py:148-160`）。因此：

- Sage 节点放在 loader 后、sampler 前，只能确保它有机会 patch 模型，不能改变 H3 传入的 `low_precision_attention=False`；
- sampler 必须连接 Sage 节点的 MODEL 输出；旁路连接原模型的分支不会获得该 clone 上的 patch；
- 普通 `ModelPatcher.clone()` 会保留 `model_options`，但后续节点若写同一个 `optimized_attention_override` 键，可能覆盖 Sage；
- 当前 SolAttn 是显式组合而非简单覆盖：`ComfyUI-SolAttn_triton/__init__.py:371-375` 保存已有 override，第 403-408 行安装组合 wrapper，并在 Sol 不适用时委托已有实现。但委托回 KJ Sage 后，H3 的精度 gate 仍会让它回退 SDPA。

### [高] 三条 SageAttention 接入路径（如何"添加 attention 节点"）
| 路径 | 方式 | 是否被 `low_precision_attention=False` 拦 |
|---|---|---|
| 1. 内置 flag | `pip install sageattention` + 启动加 `--use-sage-attention` → 走 `attention.py` 的 `attention_sage` 分发 | **是** |
| 2. 节点补丁 | 装 `kijai/ComfyUI-KJNodes`，用 `PathchSageAttentionKJ` 节点（`KJNodes/experimental`），patch `optimized_attention_override` | **是** |
| 3. SageAttention3 | ComfyUI **内置** `attention3_sage`，或其他独立 Sage3 扩展 | 内置实现**不检查**该参数；但当前 KJNodes 的 `sageattn3*` 模式仍共用第 62 行 gate，**会被拦** |

`PathchSageAttentionKJ` 的 sageattn 模式（`model_optimization_nodes.py:26`）：
`disabled / auto / sageattn_qk_int8_pv_fp16_cuda / sageattn_qk_int8_pv_fp16_triton / sageattn_qk_int8_pv_fp8_cuda / sageattn_qk_int8_pv_fp8_cuda++ / sageattn3 / sageattn3_per_block_mean`。

### [中] ComfyUI 内置 SageAttention3 可绕过门，但 KJNodes 的 Sage3 模式不行
- 内置 `attention3_sage`（`attention.py:603-615`）的回退条件只有：非 CUDA、dtype 非 fp16/bf16、有 mask、形状不匹配 —— **不检查 `low_precision_attention`**。
- 因此 H3（bf16、head_dim 128）理论上会被 `sageattn3_blackwell` 命中。
- 但 KJNodes 的 `sageattn3` / `sageattn3_per_block_mean` 只是更换第 50-55 行的底层 `sage_func`，外层仍统一经过第 60-63 行 `attention_sage` gate；对当前 H3 仍会回退 SDPA。旧版报告把 KJ Sage3 与 ComfyUI 内置 `attention3_sage` 等同，现已更正。
- 内置 Sage3 仍**无视了 H3 显式要求的高精度注意力**，输出质量有回退风险（可能重踩"乱码/质量劣化"的坑）。
- 本机 5060 Ti（sm_120/Blackwell）跑 sageattn2 已实测 sm_120 兼容（max diff 0.006），但 sageattn3 的 cu130/sm_120 Windows wheel 是否可装、装后质量如何，尚未实测。

### [中] 项目现状：sage 装了但没用，SolAttn 才是真正生效的
- `sageattention 2.2.0+cu130torch2.10.0andhigher.post6` 已装在 `python_embeded`（冒烟测试通过），但**启动不带 `--use-sage-attention`，且 H3 被 `low_precision_attention=False` 锁死** → 实际零 sage 内核。
- `triton-windows 3.7.1.post27` 也在（SolAttn 编译修复的痕迹）。
- 真正加速 H3 attention 的是 `kijai/ComfyUI-SolAttn_triton`（免训练稀疏注意力 + Morton hook），attention 7.7s→~4s/步。

---

## 项目文档审查

| 文档/源码 | 状态 | 相关结论 |
|---|---|---|
| `docs/research-sageattn2-h3.md` | RELIABLE | 已证 sage2 对 H3 必然回退 SDPA（torch.profiler 实测 sage 内核 = 0）；sage3 是唯一理论路径 |
| `docs/research-pysssss-h3-workflows.md` | RELIABLE | 工作流含 `PathchSageAttentionKJ`（Sage 节点）但 SolAttn 靠后实际生效 |
| `docs.md` | RELIABLE | SolAttn 成功加速 H3 的记录（阶段 6） |
| `comfy/ldm/minimax/model.py:181-182` | RELIABLE（权威） | H3 传 `low_precision_attention=False` 的直接证据 |
| `comfy/ldm/modules/attention.py:24-41,505-615` | RELIABLE（权威） | `attention_sage`/`attention3_sage` 回退逻辑、`sageattn`/`sageattn3` import 与 flag 处理 |
| `example/ComfyUI-KJNodes/nodes/model_optimization_nodes.py:26-136` | RELIABLE（权威） | `PathchSageAttentionKJ` 节点定义、sage 模式、同样的 `low_precision_attention` 门、`optimized_attention_override` 补丁 |

---

## 来源

- 本地权威（可逐行复核）：上述 `comfy/ldm/modules/attention.py`、`comfy/ldm/minimax/model.py`、`example/ComfyUI-KJNodes/nodes/model_optimization_nodes.py`。
- `kijai/ComfyUI-KJNodes`：`https://github.com/kijai/ComfyUI-KJNodes`（`nodes/model_optimization_nodes.py` 定义 `PathchSageAttentionKJ`）。
- `kijai/ComfyUI-SolAttn_triton`：`https://github.com/kijai/ComfyUI-SolAttn_triton`（Sol-Attn，本机实际生效的 attention 加速）。
- SageAttention Windows wheel（2.2.0，sm_120/RTX 50）：`https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows`。
- ComfyUI-SageAttention3 扩展：`https://github.com/wallen0322/ComfyUI-SageAttention3`。
- SageAttention-for-windows 分支：`https://github.com/sdbds/SageAttention-for-windows`。

> 注：以上 GitHub/网页 URL 来自项目既有文档（`research-sageattn2-h3.md` / `research-pysssss-h3-workflows.md`），本次未逐条在线复核。

---

## 矛盾与缺口

- **Perplexity MCP 本次仍未返回可核验的源码 URL/行号**，只能提供一般性判断。本次结论全部由**本地源码、运行日志与 profiler 结果**交叉验证；网络侧上游最新提交是否改变行为仍需按具体版本复核。
- **SageAttention3 质量风险未实测**：`low_precision_attention=False` 是 H3 移植者的显式选择，但未在 H3 上对比过 sage3 vs SDPA 的生成质量。
- **sageattn3 的 cu130/sm_120 Windows wheel 是否可装、性能如何**未验证。

---

## 解锁 SageAttention2 的正确做法（补充，2026-08-13）

上述"锁死"措辞不够准确——门是**可移除**的。要解锁 sage2，需**同时满足两个条件**：

1. **sage 处于活跃 attention 路径**：`attention.py:756-760` 在 import 时按 flag 给 `optimized_attention` 赋值；只有 `--use-sage-attention`（或 KJNodes 的 `PathchSageAttentionKJ` 节点 patch `optimized_attention_override`）才会让 `optimized_attention = attention_sage`。本项目启动**不带该 flag**，默认是 `attention_pytorch`，光删 kwarg 不会让 sage 被调到。
2. **绕过 `low_precision_attention=False` 门**：三种方式——
   - **(a) 改源码**：`comfy/ldm/minimax/model.py:181-182` 删掉 `low_precision_attention=False`。
   - **(b) monkey-patch**：包 `optimized_attention` 而非 `attention_sage`。因为 `minimax/model.py:27` 是 `from ... import optimized_attention`（绑了独立引用），且 `attention.optimized_attention = attention_sage` 是 import 时的值赋值，重绑 `attention_sage` 名字不会回溯；`@wrap_attn` 的 wrapper 闭包里的 `func` 从外部也够不到。正确 patch 要同时重绑两处：
     ```python
     from comfy.ldm.modules import attention as A
     import comfy.ldm.minimax.model as H3
     _orig = A.optimized_attention
     def _unlock(q, k, v, heads, *args, **kwargs):
         kwargs.pop("low_precision_attention", None)
         return _orig(q, k, v, heads, *args, **kwargs)
     A.optimized_attention = _unlock
     H3.optimized_attention = _unlock   # H3 实际调的是这条引用
     ```
   - **(c) SageAttention3**：`attention3_sage` 本身就不判这个开关（见关键发现 [中]）。

**质量提醒**：H3 作者显式关低精度注意力是主动选择（INT8/INT4 量化在该模型上大概率有可见回退），强行 unlock 是拿高精度换速度，务必逐帧对比生成质量，别只看速度。

---

## 建议（可操作下一步）

1. **默认别动 SageAttention2**：门是能拆，但拆了等于无视 H3 作者的高精度要求，收益要拿质量去赌。若非要试，按上节"解锁做法"做，并逐帧对比质量。
2. **想试 SageAttention3**：不要把当前 KJNodes 的 `sageattn3*` 模式当成绕门方案；应核对并使用 ComfyUI 内置 `attention3_sage` 路径或独立 Sage3 扩展。先备份 `python_embeded`，装完用 profiler 确认 kernel，再对比生成帧质量，并确认 sm_120/cu130 wheel 可用。
3. **本机真正有效的 attention 加速是 SolAttn**（`kijai/ComfyUI-SolAttn_triton`），已实测 attention ↓55%、20 步 ↓26%。继续用 SolAttn 即可，不必在 Sage 上纠结。
4. 若要判断 attention 是否仍是主瓶颈，先跑 `torch.profiler` / `nvidia-smi` 量化占比，再决定是否上 sage3。

---

## 搜索覆盖

- **本地（权威）**：`comfy/ldm/modules/attention.py` 的 sage 分发与 flag 处理；`comfy/ldm/minimax/model.py` 的 `low_precision_attention=False` 调用点；`example/ComfyUI-KJNodes` 的 `PathchSageAttentionKJ` 节点定义与补丁机制；项目三份 docs 交叉验证。
- **Perplexity**：2026-08-24 再次以 pro 模式检索节点日志、fallback、override 与 profiler 判据，仍未返回可核验源码引用，因此不作为关键结论依据。
- **未覆盖**：sageattn3 Windows/cu130/sm_120 实测安装与基准；GitHub 页面直读（本网络被策略拦）；H3 attention 占比的 profiler 数据（沿用历史结论）。
