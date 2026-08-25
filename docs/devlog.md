# Devlog:Minimax-H3 注意力/文本编码器加速 研究·实测·踩坑全记录

**日期**: 2026-08-13 → 2026-08-14
**作者**: Claude Code (deepseek-v4-pro) + 用户协作
**范围**: 从"为什么只看到 SolAttn 看不到 sage2"出发,一路走到:SageAttention2 解锁实测→硬崩 GPU→静态根因分析→TE 加速调研→nvfp4 实测 44× 提速。

---

## 一、这条线索是怎么一步步展开的

起点是用户一句疑问:**"are we really use sage attn2? but why I JUST SEE solattn"**。随后的对话像剥洋葱一样,每层都推翻或修正了上一层:

```
1. "sage2 被锁死,没用"        ← 初判(基于代码)
2. "其实能解锁,拆掉门就行"     ← 用户纠正
3. "解锁后加速 68% 但崩 GPU"   ← 实测
4. "为什么崩?sm120 无专属内核" ← 静态根因
5. "TE 呢?nvfp4 呢?"          ← 话题转向
6. "nvfp4 快 44×,质量无损"     ← 实测
```

---

## 二、时间线(所有动作)

### 阶段 A:研究 SageAttention2 为何不生效

| # | 动作 | 结果 |
|---|---|---|
| 1 | Grep `sage`/`solattn` 全仓库 + 读 `docs/research-sageattn2-h3.md`、`docs.md` | 确认 sage2 装了但没被用,根因 `low_precision_attention=False` |
| 2 | `/perplexity-research`(sage2 why not use + how to add node) | 写 `docs/research-sageattn-node-integration.md`;发现 KJNodes `PathchSageAttentionKJ` 节点带着**同一个门**,对 H3 同样无效 |
| 3 | 读 ComfyUI `attention.py`、`minimax/model.py`、KJNodes `model_optimization_nodes.py` 源码 | 逐行确认分发逻辑:`attention_sage` 首行判 `low_precision_attention` |

**用户纠正**:用户指出"直接改 model.py:181-182 删掉 `low_precision_attention=False` 即可解锁,不想改源码就 monkey-patch"。

| # | 动作 | 结果 |
|---|---|---|
| 4 | 验证 monkey-patch 可行性 | **发现引用链陷阱**:`minimax/model.py:27` 是 `from ... import optimized_attention`(绑独立引用),且 `attention.optimized_attention = attention_sage` 是 import 时的值赋值 → patch `attention_sage` 名字**不生效**,必须 patch `optimized_attention`(且重绑两处命名空间)。修正了用户的 patch 目标 |
| 5 | 更新报告,写入"解锁正确做法" | `docs/research-sageattn-node-integration.md` 补了可解锁的 monkey-patch 代码 |

### 阶段 B:sage2 解锁实测(用户要求"测试对比")

用户:"你测试一下对比一下开启和不开启...三个版本 sol+attn2, attn2, solattn和原版已有数据不用重跑"。

| # | 动作 | 结果 |
|---|---|---|
| 6 | EnterPlanMode 出计划,获批 | 计划:编辑 model.py + `--use-sage-attention` flag,跑 attn2(1步profile+20步) |
| 7 | `Edit` model.py:181-182 删 `low_precision_attention=False`;建 `run_sage_test.bat` | 解锁 |
| 8 | 后台启动 ComfyUI `--use-sage-attention` | 日志 `Using sage attention` 确认 flag 生效 |
| 9 | 提交 `h3_profile.json`(1步 H3SampleProfiler) | **profile 显示 sageattn 内核跑了**:`sageattention_qattn_sm89::qk_int8_sv_f8_accum` 2474ms × 52,对比 baseline cuDNN flash ~7702ms → **↓68%** |
| 10 | 提交 `h3_fp8.json`(20步) | **第 3 步崩溃**:`Fatal Python error: Aborted`,GPU 丢失 |

### 阶段 C:崩溃根因(不重跑,静态分析)

用户:"不是为什么崩溃啊"——不要继续跑,查为什么崩。

| # | 动作 | 结果 |
|---|---|---|
| 11 | 立即 `git checkout` model.py + 删 bat(环境还原) | 安全第一 |
| 12 | 读 sageattention 包源码 `core.py` | **根因铁证**:`core.py:171-178` 对 sm120 无条件走 `sageattn_qk_int8_pv_fp8_cuda`,注释明写"triton kernel not usable on sm120";fp8_cuda 内部调 `_qattn_sm89` 模块 |
| 13 | `ls` 包目录 | 只有 `_qattn_sm80/_sm89/_sm90.pyd`,**没有 sm100/sm120 模块** |
| 14 | 写根因进 `docs/research-sageattn2-h3.md` | 结论:sm89 内核经 PTX 前向兼容到 sm120,多步后非法内存访问→原生 abort→GPU reset |

### 阶段 D:TE 加速 + nvfp4(话题转向)

| # | 动作 | 结果 |
|---|---|---|
| 15 | `/perplexity-research`(TE 加速模块) | `docs/research-h3-te-acceleration.md`:不存在独立 TE 加速节点;TE 单次运行对采样贡献≈0 |
| 16 | **MCP 掉线**诊断 | 见"困难 3" |
| 17 | `/perplexity-research`(RTX50 原生 nvfp4) | `docs/research-h3-nvfp4-rtx50.md`:RTX50 原生 FP4,ComfyUI nvfp4 需 CUDA13(本机满足) |
| 18 | 完整性扫描 nvfp4 TE 文件 | 写 `check_integrity.py`;2054 张量 15.69GB **0 损坏** |
| 19 | **测速 nvfp4**(用户:"测速nvfp4") | 见"实测结果" |
| 20 | 抓原始数据(用户:"原始数据呢") | 提取日志原始行 |

---

## 三、核心实测结果

### sage2 解锁(1步 profile)
| attention 内核 | baseline | sage2 解锁 |
|---|---|---|
| 自耗时 | cuDNN flash ~7702ms | **sageattn `qk_int8_sv_f8_accum` 2474ms(↓68%)** |

### sage2 20步:崩溃
```
第 3 步采样: Fatal Python error: Aborted
崩溃栈顶: attention.py:592 (attention_sage 里 sageattn() 之后的 reshape)
nvidia-smi → "GPU is lost. Reboot the system to recover this GPU"
```

### nvfp4 TE 加载+编码(受控单测)
| TE | staged 显存 | 加载+编码 |
|---|---|---|
| **nvfp4_awq** | 14956 MB | **5.79 s** |
| int8_convrot | 25882 MB | **256.76 s** |

**nvfp4 快 ~44×、省 ~11GB。** 20步全流程 12:57,质量健康(std 49-65、唯一色 41-53k、暖调、有运动),历史离群值未复现。

---

## 四、困难与应对(重点,详细)

### 困难 1:"锁死"结论被打脸
**问题**:我初判 sage2 对 H3"必然无效",措辞用了"锁死"。
**应对**:用户纠正"能解锁,拆门即可"。我**不辩解**,重新读源码验证,发现用户对、我的措辞错。进而发现一个**更深的技术细节**——monkey-patch 的目标不是 `attention_sage` 而是 `optimized_attention`(引用链陷阱),把这个修正一并写进文档。
**教训**:把"机制上被禁用"说成"物理锁死"是过度断言。门是**可移除**的,要区分"默认回退"和"不可绕过"。

### 困难 2:sage2 实测把 GPU 崩了(最严重)
**问题**:20步测试第 3 步 `Fatal Python error: Aborted`,CUDA 驱动崩,`nvidia-smi` 报"GPU is lost, reboot to recover"。GPU 完全不可用。
**应对**:
1. **先止损**:立即 `git checkout` 还原 model.py、删测试 bat,确保环境安全。
2. **明确告诉用户 GPU 需要重启**——这是不可逆的硬件状态,不能瞒报。
3. **换思路根因**:GPU 崩了没法再跑,改走**静态代码分析**——读 sageattention 包的 `core.py` 分发逻辑 + 目录结构,发现 sm120 无专属模块、被强制塞进 sm89 fp8-V 内核。这不需要 GPU,结论照样硬。
4. 结论:不是"质量回退",是**稳定性灾难**;`try/except Exception` 兜不住 native abort。
**教训**:运行会崩溃硬件的内核前,应先查清楚该内核是否为当前架构编译(`ls` 一下包目录就知道有没有 sm120 模块)。这次是先踩了再查。

### 困难 3:Perplexity MCP 中途掉线
**问题**:`mcp__perplexity__search` 报 `No such tool available`,但 `claude mcp list` 显示 `Connected`。
**应对**(按 skill 的失败流程,不瞎改):
1. `WaitForMcpServers` → `Failed to connect`(确认会话级)。
2. `claude mcp get perplexity` → 配置对(cookie 路径、代理都对)。
3. 运行时 `fastmcp import` 检查 → OK。
4. 手动 spawn 服务器进程 → exit 0,干净启动。
5. **结论:服务器健康,是会话的 MCP 客户端进程没连上**,需要用户 `/mcp` reconnect 或重启。
6. 按 skill 约定停止探测,明确告诉用户重连操作。
**教训**:CLI 显示 "Connected" 不代表当前会话工具可用;区分"配置层连接"和"会话层工具加载"。不反复重试已断的工具。

### 困难 4:Perplexity 占位式回答
**问题**:多次返回 "I don't have live access... 可以帮你查" 的提纲式占位,无实质 URL/代码细节。
**应对**:不盲信;把网络侧标为"低置信";**用本地源码/日志做权威证据**(源码逐行可复核);对实质缺口做一次合并 follow-up;报告里明确区分"本地权威"vs"web 未证实"。
**教训**:占位回答 ≠ 失败,但要显式标注,别把"泛泛断言"当结论(项目文档 `research-pysssss-h3-workflows.md` 里也记录过同样的 Perplexity 降级问题)。

### 困难 5:ComfyUI API 工作流验证失败
**问题**:TE 单测工作流提交报 `400 prompt_no_outputs`,加 `PreviewConditioning` 后又报 `missing_node_type: PreviewConditioning not found`。
**应对**:
1. `prompt_no_outputs` → 需至少一个终端节点消费结果。
2. 查 `object_info` API 找可用节点 → `PreviewConditioning` 本机没注册。
3. 找到 **`PreviewAny`**(接受 `source: ANY` 类型)→ 用它做终端节点,单测跑通。
**教训**:工作流必须有输出消费端;节点名以 `object_info` 实际注册为准,别猜节点名。

### 困难 6:nvfp4 历史污名
**问题**:项目历史记录 nvfp4 TE 曾"下载损坏 + 固定离群值(max=15974)",web 又说"nvfp4 更快",矛盾。
**应对**:不轻信任何一方,分两步验证:
1. **先验文件完整性**(写 `check_integrity.py`,逐张量零区扫描)→ 0 损坏,排除"损坏残留"。
2. **再实测**加载+编码速度 + 20步质量 → 快 44×,质量健康,离群值未复现。
**结论**:旧离群值 = 下载损坏残留,不是 nvfp4 量化伪影。**教训**:对"有前科"的东西,用数据洗白/坐实,而不是靠嘴。

---

## 五、写入的记忆(跨会话持久)

| 记忆文件 | 一句话 |
|---|---|
| `int8-model-used-sage-attn2.md` | 用户回忆:int8 变体曾真跑到 sage2(与 fp8 不同),待验证与代码张力 |
| `sageattn2-crashes-h3-gpu.md` | 解锁 sage2 加速 68% 但第 3 步硬崩 GPU,**不要给 H3 开 sage2** |
| `nvfp4-te-44x-faster-on-rtx50.md` | nvfp4 TE 比 int8 快 44×、省 11GB、质量干净;旧离群值是损坏残留 |

(另有既存的 `triton-cuda-utils-fix`、`solattn-h3-motion-context-coexist`、`use-python-download-clients`)

---

## 六、本会话创建/修改的文件

### 新建文档(4)
- `docs/research-sageattn-node-integration.md` — sage2 为何不生效 + 如何加 attention 节点 + 解锁做法
- `docs/research-h3-te-acceleration.md` — TE 加速:无独立节点,单次运行贡献≈0
- `docs/research-h3-nvfp4-rtx50.md` — RTX50 原生 nvfp4 + 换 TE 建议 + 实测表
- `docs/devlog.md` — 本文

### 修改文档(1)
- `docs/research-sageattn2-h3.md` — 追加"实测解锁 sage2 加速 68% 但崩 GPU"+"崩溃根因"

### 新建工具脚本(4)
- `comfyui_download/submit_prompt.py` — 通用 API 提交+轮询
- `comfyui_download/check_integrity.py` — safetensors 逐张量零区完整性扫描
- `comfyui_download/quality_check.py` — mp4 抽帧质量检查(std/唯一色/运动)
- `comfyui_download/te_bench_nvfp4.json` / `te_bench_int8.json` / `h3_fp8_nvfp4te.json` — 测速工作流

### 新建记忆(3)
- 见上表

---

## 七、当前结论(截至 2026-08-14)

1. **SageAttention2 对 H3 不可用**:解锁后能加速 attention 68%,但 sm120 无专属内核、被塞进 sm89 fp8-V 内核,多步硬崩 GPU。**H3 作者的 `low_precision_attention=False` 是防崩溃的,不是精度洁癖。**
2. **SolAttn 仍是 H3 最快且稳定的采样加速**(attention ↓55%,20步 ↓26%);2026-08-24 完整生成 A/B 进一步确认其冷启动也比 FlashAttention2 快。
3. **TE 换 nvfp4 是明确优化**:加载从 256s→6s、省 11GB、质量无损。与 SolAttn 叠加 = 当前最优组合。
4. **最快出片**:热启动 nvfp4+SolAttn+EasyCache ≈ **3:15**(init 0:15 + 采样 2:43 + 解码 0:17);冷启动 10:48。详见第八节实测。

---

## 八、最优组合实测:冷启动 vs 热启动(nvfp4 + SolAttn + EasyCache)

**2026-08-14 受控实测**(864×480/124帧/20步,冷 seed 42 / 热 seed 43)。把第七节"最优组合 ~10.5min 🔮"里的推算项换成同会话实测。

| 阶段 | 冷启动(seed 42) | 热启动(seed 43) | 节省 |
|---|---|---|---|
| 模型初始化(TE+fl2va) | 3:50 | 0:15 | −3:35 |
| 采样 20 步(SolAttn+EasyCache) | 2:47 | 2:43 | −0:04 |
| 解码(VAE+编码) | 4:11 | 0:17 | −3:54 |
| **总耗时** | **10:48** | **3:15** | **−7:33** |

- 配置:SolAttn tau=1.3 / exact_kv_and_rows / morton 2d_frame / int8_qk;EasyCache 0.3/0.2/0.9;nvfp4 TE(14956MB staged)。两次均 `skipped 7/20 steps (1.54x speedup)`。
- 质量两条都健康:std 42.9–58.1、唯一色 37k–66k、暖调 R>G>B、运动随间隔递增(10→24/255);无 EasyCache 伪影、无离群值。
- **热启动 3:15 里采样占 84%**,init/decode 几乎归零——fl2va + 两个 VAE 都留在显存(`0 models unloaded`)。
- **踩坑:执行缓存 ≠ 热启动**。第一次"热启动"复用 seed 42,ComfyUI 执行缓存命中 `3.24s` 返回(采样没真跑)。换 seed 43 强制重采样,才拿到真 3:15。
- **资源占用**:满载时系统内存 ~53.8GB(触发压缩)+ VRAM 16GB 占满;跑完停服即释放。
- 结论:最快出片 = 热启动 nvfp4+SolAttn+EasyCache ≈ **3:15**;比此前"≈7-8 分钟"估算更低——因为热启动连 decode 也从 4:11 压到 0:17(VAE 常驻)。

---

## 九、FlashAttention2 完整生成对照（2026-08-24）

为验证 KJNodes 的 `Patch Flash Attention KJ` 是否比现有 SolAttn 更快，创建独立 portable 测试环境：Python 3.13.14、Torch 2.11.0+cu130、FlashAttention 2.9.0；模型目录通过 junction 与原环境共用，原 Torch 2.13 环境未改动。FlashAttention 已先在 RTX 5060 Ti（sm_120）上通过 bf16 CUDA 冒烟测试。

**受控条件**：两次均为 H3 FL2VA FP8 + Qwen3VL 32B NVFP4 AWQ TE + EasyCache 0.3/0.2/0.9，864×480、124 帧、20 步、seed 42、冷启动并包含模型初始化、采样、双 VAE 解码和 MP4 保存。唯一主要 attention 差异为 SolAttn 与 FlashAttention2；Torch 版本分别为 2.13/cu130 和 2.11/cu130。

| 阶段 | SolAttn + EasyCache | FlashAttention2 + EasyCache | Flash 相对 Sol |
|---|---:|---:|---:|
| 模型初始化（TE+fl2va） | 3:50 | 3:56 | +0:06（+2.6%） |
| 采样 20 步 | 2:47 | 3:12 | **+0:25（+15.0%）** |
| 解码、编码及其他 | 4:11 | 4:19 | +0:08（+3.2%） |
| **冷启动总耗时** | **10:48** | **11:27** | **+0:39（+6.0%，更慢）** |
| EasyCache 跳步 | 7/20（1.54×） | 8/20（1.67×） | Flash 甚至多跳 1 步 |

- FlashAttention 路由确认：`run_flashattn_full.log:72` 输出 `Using flash attention 2: cast_dtype=torch.bfloat16`；该 KJ patch 不受 H3 的 `low_precision_attention=False` gate 限制。
- 完整生成成功：`run_flashattn_full.log:88` 为 `Prompt executed in 00:11:27`，输出 `MiniMax_H3_nvfp4_flashattn_easycache_00001_.mp4`，没有 CUDA、DLL 或 FlashAttention 异常。
- SolAttn 对照：`run_nvfp4_sol_easycache_cold.log:81-87`，冷初始化 3:50、采样 2:47、总耗时 10:48。
- 纯 kernel 微基准中 FlashAttention2 相对 Torch SDPA 为 2.38×：H3 序列 15828 时 158ms vs 377ms，序列 18771 时 223ms vs 531ms；但这只说明它比**稠密 SDPA**快。SolAttn 是 H3 专用稀疏 attention，完整生成仍更快。
- 本次比较略微有利于 FlashAttention：它的 EasyCache 跳过 8 步，而 SolAttn 只跳过 7 步；即便如此 Flash 采样仍慢 25 秒，因此方向明确。
- Torch 2.11 与 2.13 的同形状 SDPA 基线几乎相同（约 377/531ms），没有证据表明结果是 Torch 降级导致的假性差异。不过严格归因仍受不同 Torch 版本影响，若以后出现完全匹配 Torch 2.13/cu130/cp313/sm120 的 FlashAttention wheel，应再复测。

**结论**：FlashAttention2 在 H3 上确实生效且稳定完成一次 20 步生成，但它替代的是稠密 attention；相较 H3 专用 SolAttn，冷启动总耗时慢约 6%，采样段慢约 15%。当前生产配置继续使用 **NVFP4 TE + SolAttn + EasyCache**；FlashAttention 测试副本保留用于兼容性和后续 wheel 复测，不替换主环境。

---

## 十、FlashAttention2 + SolAttn + EasyCache 冷热启动（2026-08-24）

在 Torch 2.11/cu130/FlashAttention 2.9.0 测试副本中串联：

`UNET → Patch Flash Attention KJ → SolAttnPatch → EasyCache`

两次均为 H3 FL2VA FP8 + Qwen3VL 32B NVFP4 AWQ TE、864×480、124 帧、20 步；冷启动 seed 42，保持同一 ComfyUI 进程后改为 seed 43 做真热启动。日志确认 `Using flash attention 2`，随后 SolAttn 检测到已有 override，并采用“SolAttn 优先、其余调用委托 FlashAttention”的组合方式（`run_flash_solattn_cold_hot.log:72-74`）。两次 EasyCache 均跳过 7/20 步（1.54×）。

| 阶段 | SolAttn + EasyCache | Flash + SolAttn + EasyCache | 组合链相对 Sol |
|---|---:|---:|---:|
| 冷初始化 | 3:50 | 3:53 | +0:03 |
| 冷采样 20 步 | 2:47 | 2:49 | +0:02 |
| 冷解码、编码及其他 | 4:11 | 3:29 | −0:42 |
| **冷启动总耗时** | **10:48** | **10:11** | **−0:37（表面快 5.7%）** |
| 热初始化 | 0:15 | 0:15 | 持平 |
| 热采样 20 步 | 2:43 | 2:46 | +0:03 |
| 热解码、编码及其他 | 0:17 | 0:16 | −0:01 |
| **热启动总耗时** | **3:14.79** | **3:16.94** | **+0:02.15（慢 1.1%）** |

API 轮询侧测得组合链冷启动 615.57 秒、热启动 200.28 秒；与 ComfyUI 内部计时存在约 4 秒的轮询/提交开销，因此性能比较采用日志中的 `10:11` 和 `196.94s`。两次均成功输出 MP4：

- `MiniMax_H3_nvfp4_flash_solattn_easycache_cold_00001_.mp4`（1.23 MB）
- `MiniMax_H3_nvfp4_flash_solattn_easycache_hot_00001_.mp4`（1.26 MB）

**解释与结论**：

- 冷启动总时间看似快 37 秒，但模型初始化慢 3 秒、采样慢 2 秒；全部表面收益来自 VAE 解码/编码与收尾阶段偶然快 42 秒。Flash patch 只装在 diffusion MODEL 上，不会直接加速独立 VAE，因此不能把这 37 秒归因于 FlashAttention。
- 热启动更能隔离 attention 差异：组合链采样慢 3 秒、总耗时慢 2.15 秒，基本可视为持平但无收益。
- H3 主 self-attention 已被 SolAttn 优先接管，FlashAttention只处理 SolAttn 拒绝/委托的其余 attention；本次结果说明这些剩余调用对总耗时贡献很小。
- 组合测试运行在 Torch 2.11，而旧 SolAttn 对照是 Torch 2.13，仍存在版本变量；但现有数据不支持为了组合 FlashAttention 而替换生产栈。

**最终推荐不变**：生产继续使用 **NVFP4 TE + SolAttn + EasyCache**。Flash + SolAttn 组合能够稳定运行，但热启动没有提速，增加了 Torch 降级与额外二进制依赖，不值得作为默认配置。

---

## 十一、Flash + SolAttn 组合重测与定向优化（2026-08-24）

上一节的组合数据与 SolAttn 基线分别来自 Torch 2.11 和 2.13，且冷启动结果受 VAE/文件缓存波动影响。为消除这些问题，本轮全部在同一个测试副本中完成：Python 3.13.14、Torch 2.11.0+cu130、FlashAttention 2.9.0、相同模型、分辨率、124 帧、20 步、seed 44/45，并以热采样时间作为主要指标。

### 11.1 同环境基线与问题定位

| 配置 | 热初始化 | 热采样 | 热总耗时 | EasyCache |
|---|---:|---:|---:|---:|
| SolAttn tau1.3 rows + EC0.30 | 0:15 | **2:44** | **195.49s** | 7/20 |
| Flash + Sol tau1.3 rows + EC0.30 | 0:15 | 2:46 | 196.94s | 7/20 |
| Flash + Sol tau1.3 exact_kv + EC0.30 | 0:16 | 2:46 | 197.61s | 7/20 |

- 同环境确认：Flash + Sol 比 Sol-only 慢约 1–2 秒；不是 Torch 2.11/2.13 差异。
- 节点顺序必须是 `UNET → Flash → SolAttn → EasyCache`。日志确认 SolAttn 优先接管 H3 长 self-attention，并把不适用路径委托 Flash。
- `exact_kv_and_rows → exact_kv` 没有带来可重复速度收益，因此 conditioning rows 不是当前墙钟瓶颈。为了保留音频 query rows 的精确 attention，默认继续用 `exact_kv_and_rows`。
- `allow_compile` 保持关闭：当前工作流没有模型 compile 节点，打开它本身不会启动编译。`use_tma` 也保持关闭，源码明确说明会增加 QKV 复制和峰值显存且尚未测得更快。

### 11.2 参数优化结果

先将 SolAttn `tau` 从 1.3 调至 1.5，再单独提高 EasyCache 阈值：

| 配置 | 热采样 | 热总耗时 | 相对同环境基线 | 跳步 |
|---|---:|---:|---:|---:|
| Sol tau1.3 rows + EC0.30 | 2:44 | 195.49s | 基线 | 7/20 |
| Flash + Sol tau1.5 exact_kv + EC0.30 | 2:43 | 193.96s | −1.53s（0.8%） | 7/20 |
| Flash + Sol tau1.5 rows + EC0.30 | 2:44 | 195.15s | −0.34s（持平） | 7/20 |
| **Flash + Sol tau1.5 rows + EC0.35** | **2:34** | **185.47s** | **−10.02s（5.1%）** | **8/20** |

优化后完整参数：

- FlashAttention2：KJ patch，bf16，`allow_compile=false`
- SolAttn：`tau=1.5`、`start=0.2`、`end=0.9`、`min_tokens=4096`、`int8_qk=true`
- 质量保护：`sink_conditioning=exact_kv_and_rows`、Morton `2d_frame`
- EasyCache：`reuse_threshold=0.35`、`start=0.2`、`end=0.9`
- `use_tma=false`

工作流：`comfyui_download/h3_fp8_nvfp4te_flash_solattn_tau15_rows_easycache035.json`。

### 11.3 冷启动解释

本轮进程冷启动总耗时分布为约 9:32–10:17，但测试按顺序重复读取同一批 15–20GB 模型，Windows 文件缓存逐轮变热，因此后面的所谓“冷进程”并非严格的磁盘冷启动，不能拿 9:32 对 10:16 宣称 attention 优化节省了 44 秒。可复现的主结论只采用同一进程第二次生成：优化后热总耗时 185.47 秒，较同环境基线 195.49 秒快 10.02 秒。

### 11.4 输出与质量检查

优化候选冷热两次都成功完成，没有 CUDA、FlashAttention、SolAttn 或 DLL 错误。热输出：

`MiniMax_H3_flash_solattn_tau15_rows_ec035_hot_00001_.mp4`

媒体检查：124 帧、864×480、24 FPS、5.167 秒；H.264 视频 + AAC 32kHz 双声道音频均完整。关键帧拼图检查未见断帧、结构崩坏、主体消失或明显 EasyCache 拖影；狗的运动、草地、逆光和背景连续。

但加速不是数学无损：相同 seed 下，EC0.35 对 EC0.30 的平均像素绝对差为 8.415/255、平均 PSNR 24.33dB；音频相关系数 0.886、相对 SNR 6.55dB。它们语义和结构相近，但细节轨迹与音频波形确实改变。对比拼图保存于 `comfyui_download/flash_solattn_optimization_contact_sheet.jpg`。

### 11.5 最终结论

- **稳定保守配置**仍是 NVFP4 TE + SolAttn tau1.3 rows + EasyCache 0.30；不需要 Flash，热耗时约 3:15。
- **速度候选配置**是 NVFP4 TE + FlashAttention2 + SolAttn tau1.5 rows + EasyCache 0.35；本次热耗时 **3:05.47**，快约 10 秒（5.1%）。
- 可确认的主要收益来自 EasyCache 多跳过 1 步，而不是 FlashAttention；Flash 主要作为 SolAttn dense fallback，单独贡献接近零甚至略有开销。
- 因为当前只测试两个 seed 且数值输出有明显变化，速度候选暂不覆盖主生产工作流。建议先人工听看 seed 45 输出，再用更多 prompt/seed 做质量回归；若质量可接受，再把 EC0.35 作为快速预览档。

---

## 十二、把本机 ComfyUI 转发到云端公网（2026-08-24）

目标：把云端服务器 `106.55.30.150` 的 80 端口从 New API 切到**本机 ComfyUI（生产环境）**，并用 HTTPS + Basic Auth 保护。

### 12.1 总体链路

```
公网 → 106.55.30.150:552 (Caddy, HTTPS video.geekq.xyz, Basic Auth)
        ↓ reverse_proxy
云端 127.0.0.1:8188 (sshd 反向映射端口)
        ↓ SSH -R 反向隧道
本机 127.0.0.1:8188 (ComfyUI 0.30.0, Torch 2.13)
```

### 12.2 SSH 反向隧道与自启动

- 脚本：`F:\python\llamacpp\deploy-cloud\start-comfyui-cloud.bat`
  - 本机 8188 未监听则先启 ComfyUI，再进入 `-R 127.0.0.1:8188:127.0.0.1:8188 root@106.55.30.150` 的 watchdog 循环。
- 登录自启动任务：`ComfyUI Cloud Tunnel`。
- 运行态确认：`cmd(26504)` 看门狗 + `ssh(21204)` 隧道进程 + 计划任务 `Ready`。

### 12.3 云端 Caddy 切换（80 → ComfyUI）

- 原配置：`:80 → 127.0.0.1:3000 (New API)`。
- 切换后：`video.geekq.xyz:552 → 127.0.0.1:8188` + `basic_auth`（用户名 `comfyui`）。
- 备份：`/etc/caddy/Caddyfile.before-comfyui`、`Caddyfile.comfyui-http-basic`、`Caddyfile.before-https-552`。
- `new-api.service` 已 `stop && disable`（inactive/disabled）。
- 认证凭据（随机密码）：`F:\python\llamacpp\deploy-cloud\comfyui-cloud-auth.txt`。
- 一键回滚 New API：`F:\python\llamacpp\deploy-cloud\restore-newapi-cloud.bat`。

### 12.4 证书（video.geekq.xyz）

- 来源：`F:\python\h3\models\video.geekq.xyz_other\video.geekq.xyz_other\`（`_bundle.crt`/`.pem`/`.key`/`.csr`）。
- 校验：`CN=video.geekq.xyz`、`SAN=DNS:video.geekq.xyz`、`issuer=TrustAsia DV TLS RSA CA 2024`、有效期 `2026-08-24 → 2026-11-22`、证书与私钥 sha256 一致。
- 上传至 `/etc/caddy/certs/`（`chown root:caddy` + `chmod 640`），Caddy 手动 `tls` 配置。
- **443 被腾讯云安全组拦截**：云主机 `tcpdump` 收不到任何外部 443 SYN；证书在云主机本地 TLS 验证却是 OK（TLS1.3 verify=0）。Caddy/证书/Linux 防火墙均无误。
- 尝试 `sslip.io`/`nip.io` 免费自动 HTTPS 失败（腾讯云 DNSPod 拦截 + 本机 DNS 污染 `198.18.x`）。因此**改用 552 端口**。

### 12.5 端口 552

- Caddy 监听 `*:552`，已关闭 443。
- 验证：`openssl s_client` TLS1.3 `Verify return code: 0`；认证 API `200`；WSS `101 Switching Protocols`。
- 本机浏览器打不开的根因是代理 Fake-IP（非云端）：Vortex 把 `video.geekq.xyz` 解析成 `198.18.0.143`。

### 12.6 本地代理 Fake-IP 处理（Vortex / Bitz Net）

- 配置文件：`C:\Users\SHUAIBI\.config\com.vortex.helper\config.yaml`（改动前备份 `.before-video-geekq`）。
- 新增三处：
  - `hosts:  video.geekq.xyz: 106.55.30.150`
  - `dns.fake-ip-filter: - video.geekq.xyz`
  - `rules: - DOMAIN,video.geekq.xyz,DIRECT`
- 用 **Python UTF-8** 精确改回 YAML（此前 PowerShell 把中文节点名写成乱码导致校验失败），`com.vortex.helper.exe -t` 校验通过。
- 通过本地控制端口 `127.0.0.1:39798` `PUT /configs?force=true` 热重载，返回 `204`。
- 因 TUN/`use-hosts true` 生效，Chrome 旧进程仍缓存 Fake-IP，需**完全退出重开浏览器**。

### 12.7 DNS（video.geekq.xyz 解析问题）

- 初始 `video.geekq.xyz` 为 **NXDOMAIN**，原因：子域悬挂委派。
- 权威记录：
  - `geekq.xyz` NS = `flowers/humid.dnspod.net`（父域，正常）
  - `video.geekq.xyz` NS = `f1g1ns1/f1g1ns2.dnspod.net`（子域委派）
- 关键判据：向 `f1g1ns1` 查 `video A` 返回的是**父区 SOA（humid.dnspod.net）**，而非子区自己的 SOA → 该 NS 实际未托管此子区，属 dead delegation。
- 结论（DNSPod 面板操作）：在 `geekq.xyz` 区删除 `video` 的 2 条 NS 委派，再加 `video A → 106.55.30.150`；**顶层 `geekq.xyz` 的 NS（flowers/humid）绝不能动**，否则整站不解析。

### 12.8 502 排障（最新）

- 症状：DNS 生效后 `https://video.geekq.xyz:552/` 返回 **502**。
- 根因：本机 ComfyUI（8188）进程消失；SSH 隧道虽在，但 Caddy→本机 8188 时 `connection reset by peer`（Caddy 日志反复出现）。
- 诊断链：云端 552 监听 OK → 云端 loopback 8188（sshd）OK → 本机 8188 无监听 → 隧道映射到死端口即 RST。
- 处理：重启生产 ComfyUI（`Start-Process python main.py --port 8188`），等 8188 就绪后隧道自动回通。

### 12.9 残留与要点

- 云端唯一缺口：`video.geekq.xyz` 的 A 记录（删子域 2 条 NS 委派 + 加 `A → 106.55.30.150`）。
- 本机 ComfyUI 必须常驻；watchdog 只在 8188 已监听时才进入 SSH 循环，ComfyUI 若挂掉需保证能自启。
- 安全：HTTP/HTTPS + Basic Auth；密码在本地 `comfyui-cloud-auth.txt`，勿入 git。
- 回滚入口：`restore-newapi-cloud.bat`（恢复 `Caddyfile.before-comfyui` + 启用 `new-api.service`）。
