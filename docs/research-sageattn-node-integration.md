# 研究：SageAttention2 为何在 ComfyUI（H3）里不生效，以及如何添加 attention 节点

**日期**: 2026-08-13
**研究方式**: Perplexity pro（1 主搜索 + 1 follow-up，均降级为占位回答）+ 本地 ComfyUI 源码逐行核对 + 项目文档交叉验证
**问题**: 为什么 SageAttention2 在本机 ComfyUI 里没用上？如何在 ComfyUI 里添加 SageAttention 节点？

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

### [高] 三条 SageAttention 接入路径（如何"添加 attention 节点"）
| 路径 | 方式 | 是否被 `low_precision_attention=False` 拦 |
|---|---|---|
| 1. 内置 flag | `pip install sageattention` + 启动加 `--use-sage-attention` → 走 `attention.py` 的 `attention_sage` 分发 | **是** |
| 2. 节点补丁 | 装 `kijai/ComfyUI-KJNodes`，用 `PathchSageAttentionKJ` 节点（`KJNodes/experimental`），patch `optimized_attention_override` | **是** |
| 3. SageAttention3 | `sageattn3` 的 `sageattn3_blackwell`（内置 `attention3_sage`，或 KJNodes 的 `sageattn3` 模式，或 `wallen0322/ComfyUI-SageAttention3` 扩展） | **否**（只判 dtype/mask/shape，见下） |

`PathchSageAttentionKJ` 的 sageattn 模式（`model_optimization_nodes.py:26`）：
`disabled / auto / sageattn_qk_int8_pv_fp16_cuda / sageattn_qk_int8_pv_fp16_triton / sageattn_qk_int8_pv_fp8_cuda / sageattn_qk_int8_pv_fp8_cuda++ / sageattn3 / sageattn3_per_block_mean`。

### [中] 唯一能绕过门的 Sage 路径是 SageAttention3（但以质量为赌注）
- 内置 `attention3_sage`（`attention.py:603-615`）的回退条件只有：非 CUDA、dtype 非 fp16/bf16、有 mask、形状不匹配 —— **不检查 `low_precision_attention`**。
- 因此 H3（bf16、head_dim 128）理论上会被 `sageattn3_blackwell` 命中。
- 但这**无视了 H3 显式要求的高精度注意力**，输出质量有回退风险（可能重踩"乱码/质量劣化"的坑）。
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

- **Perplexity MCP 本次降级为占位回答**（1 主搜索 + 1 follow-up 均返回"无实时访问、请确认"的提纲式内容，无实质 URL、无代码级细节）。这与项目既往经验一致（`research-pysssss-h3-workflows.md:82`）。本次结论全部改由**本地源码逐行核对**得出，代码级证据可靠；网络侧的新版本/新 wheel 信息未获得。
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
2. **想试 SageAttention3**：走 KJNodes 的 `sageattn3` / `sageattn3_per_block_mean` 模式，或装 `wallen0322/ComfyUI-SageAttention3` 扩展；但先备份 `python_embeded`，装完对比生成帧质量（别只看速度），并确认 sm_120/cu130 wheel 可用。
3. **本机真正有效的 attention 加速是 SolAttn**（`kijai/ComfyUI-SolAttn_triton`），已实测 attention ↓55%、20 步 ↓26%。继续用 SolAttn 即可，不必在 Sage 上纠结。
4. 若要判断 attention 是否仍是主瓶颈，先跑 `torch.profiler` / `nvidia-smi` 量化占比，再决定是否上 sage3。

---

## 搜索覆盖

- **本地（权威）**：`comfy/ldm/modules/attention.py` 的 sage 分发与 flag 处理；`comfy/ldm/minimax/model.py` 的 `low_precision_attention=False` 调用点；`example/ComfyUI-KJNodes` 的 `PathchSageAttentionKJ` 节点定义与补丁机制；项目三份 docs 交叉验证。
- **Perplexity**：1 次主搜索（pro，5 项清单）+ 1 次 follow-up（强制要代码级细节）—— 均降级为占位回答，未获得网络侧增量。
- **未覆盖**：sageattn3 Windows/cu130/sm_120 实测安装与基准；GitHub 页面直读（本网络被策略拦）；H3 attention 占比的 profiler 数据（沿用历史结论）。
