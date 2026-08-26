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

---

## 十三、SageAttention3 Blackwell wheel 重试（2026-08-25）

目标：重试 `models/sageattn3-1.0.0%2Bcu130torch2.10.0-cp312-cp312-win_amd64.whl`。该 wheel 是 Blackwell 专属 FP4 attention，而此前 SageAttention2 在 H3 第 3 步把 GPU/驱动硬崩。

### 13.1 隔离环境与接入

- wheel 内含 `fp4attn_cuda.cp312-win_amd64.pyd` / `fp4quant_cuda.cp312-win_amd64.pyd`，只能用于 CPython 3.12；生产与 Flash 测试 portable 均是 Python 3.13，不能直接安装。
- 新建隔离环境 `ComfyUI_sage3_py312/`：Python 3.12.0、Torch 2.10.0+cu130、Triton Windows 3.6.0、ComfyUI 0.30.0、comfy-kitchen 0.2.26。模型通过 `extra_model_paths.yaml` 只读复用生产目录；所有 custom nodes 禁用，生产 portable 未修改。
- ComfyUI 的 `--use-sage-attention` 只会选择 Sage2，而且 H3 的 `low_precision_attention=False` 会让 Sage2回退 SDPA。隔离副本因此用 `COMFY_SAGE3=1` 选择内置 `attention3_sage`；不设置时可用 `--use-pytorch-cross-attention` 跑同环境 SDPA 对照。
- 独立启动入口：`ComfyUI_sage3_py312/run_sage3_test.bat`（`127.0.0.1:8191`）。测试工作流：`h3_sage3_20step.json` / `h3_sdpa_20step.json`。

### 13.2 Blackwell kernel 验证

- Torch 正确识别 RTX 5060 Ti：CUDA 13.0、compute capability `(12, 0)`。
- 合成冒烟测试均成功、BF16 输出有限值：`(1,2,1152,128)`；H3 风格 `(1,24,4096,128)` 热调用 3.67ms。
- 完整 H3 日志直接证明主 attention 命中原生 Sage3，而不是配置假阳性或 SDPA fallback：
  `SageAttention3 Blackwell kernel active: q=(1, 56, 15441, 128) dtype=torch.bfloat16`。
- 20 步全部跑完，没有第 3 步 native abort、CUDA 错误或 GPU lost。

### 13.3 同环境受控 A/B

两次均为同一 Python 3.12 / Torch 2.10 / ComfyUI 0.30 环境，FL2VA FP8 + Qwen3VL NVFP4 TE、864×480、124 帧、20 步、seed 42、无 EasyCache/SolAttn/custom nodes；唯一 attention 差异为 SDPA 与 SageAttention3。

| 配置 | 模型初始化 | 采样 20 步 | 稳态每步 | 总耗时 |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 3:43 | 5:10 | 15.54s | 12:20 |
| SageAttention3 Blackwell | 3:58 | 3:41 | 11.08s | 11:49 |
| 差值 | +0:15 | **−1:29（−28.7%）** | **−4.46s** | −0:31 |

- 可靠收益采用采样段：Sage3 让 H3 纯采样从 310s 降至 221s，约 **1.40×**。冷总耗时只快 31s，是因为文件缓存、首次 Triton JIT、模型初始化和 VAE 解码波动掩盖了采样收益。
- 旧 Torch 2.13 SDPA 基线为 5:15 @15.6s/步，与本轮 Torch 2.10 SDPA 的 5:10 @15.54s/步接近，说明提速不是 Torch 版本差异制造的假象。

### 13.4 输出与质量

- Sage3 产物：`ComfyUI_sage3_py312/output/video/MiniMax_H3_sage3_py312_torch210_20step_00001_.mp4`。
- 媒体完整：124 帧、864×480、24 FPS、5.167s；H.264 + AAC 32kHz 双声道。
- 帧统计健康：frame 10/60/120 std 56.1/67.9/65.9，唯一色 70.2k/65.9k/59.3k；运动 gap 1/5/20 为 9.28/15.71/25.17，未见静帧或塌缩。
- 联系表 `comfyui_download/sage3_sdpa_contact_sheet.jpg` 显示狗、草地、逆光、运动和 anatomy 连贯，没有明显结构崩坏。
- 但 Sage3 不是数值透明替换：同环境同 seed 对 SDPA 的平均像素绝对差 **29.65/255**、平均 PSNR **14.90dB**、帧相关 **0.719**；音频相关 0.917、相对 SNR 7.91dB。语义一致，但姿态、镜头轨迹和细节明显改变。
- Torch 2.13 SDPA 对 Torch 2.10 SDPA 仅 MAD 9.27/255、PSNR 23.12dB、相关 0.950，因此大部分轨迹变化来自 Sage3 FP4 attention，而非 Torch 版本。

### 13.5 结论

- **SageAttention3 Blackwell wheel 在 RTX 5060 Ti / H3 上真实生效且完成一次 20 步稳定生成**，与会硬崩 GPU 的 Sage2 结论不同。
- 它在不跳步的稠密 attention 基线上把采样缩短约 29%，但同 seed 输出轨迹变化很大；当前只能评为“速度候选”，不能称为无损优化。
- 生产配置继续保留 **NVFP4 TE + SolAttn + EasyCache**：现有热总耗时约 3:15，且已做更多回归。Sage3 暂不写入生产 Python 3.13 环境，也不替换 SolAttn；若要晋级，下一步应在隔离环境做多个 prompt/seed 的人工画面与音频回归，再测试 Sage3 与 EasyCache/SolAttn 的组合关系。

### 13.6 Sage3 + SolAttn + EasyCache 分阶段验证

按 FlashAttention 测试流程逐级放大，任何一层失败都不进入下一层：

1. **静态路由**：SolAttn 的 override 会先处理可稀疏调用，dense warm-up/cool-down 委托当前全局 backend；因此 `COMFY_SAGE3=1` 时会形成 `SolAttn → Sage3 fallback`。EasyCache 是 ComfyUI 内置节点，不需要加载 KJNodes。
2. **合成烟雾**：`(1,24,4096,128)` BF16 上，SolAttn INT8 pointer 稀疏内核命中，统计 `sparse=1/errors=0`；范围外 dense fallback 确认 `attention3_sage.kernel_logged=True`，两路输出均为有限值。
3. **真实 H3 4 步烟雾**：成功记录 `[sol_attn] sparse (1, 15441, 56, 128) tau=1.3 int8 pointer`，同时 Sage3 负责 dense warm-up，EasyCache wrapper 正常；4 步 52.30s，无 CUDA/native 错误。最初 1 步烟雾只在终端 `SaveLatent` 因 H3 `NestedTensor` 不支持 `.contiguous()` 报错，attention 已成功完成；改用 `PreviewAny` 后通过。
4. **20 步全流程**：双 VAE 解码、H.264/AAC 保存均成功；EasyCache 每次跳过 7/20 步（1.54×）。

配置保持生产参数：SolAttn tau 1.3、0.2–0.9、INT8 QK、`exact_kv_and_rows`、Morton `2d_frame`；EasyCache 0.30、0.2–0.9。工作流位于隔离目录 `h3_sage3_sol_easy_20step.json`。

### 13.7 三件套同环境严格 A/B

两组均在同一 Python 3.12 / Torch 2.10 / ComfyUI 0.30 环境，只改变 dense fallback：PyTorch SDPA 或 SageAttention3；SolAttn、EasyCache、模型、分辨率、20 步和 seed 完全相同。

| 配置 | 冷采样 | 冷总耗时 | 热采样 | 热总耗时 | 跳步 |
|---|---:|---:|---:|---:|---:|
| SolAttn + EasyCache（SDPA fallback） | 2:45 | 9:40.32 | 2:43 | 3:13.41 | 7/20 |
| **Sage3 + SolAttn + EasyCache** | **2:15** | **9:38.95** | **2:15** | **2:44.99** | 7/20 |
| 三件套增量 | −0:30 | −0:01.37 | **−0:28（−17.2%）** | **−0:28.42（−14.7%）** | 相同 |

- 可靠结论取同进程 seed 43 热 A/B：三件套热总耗时 **2:44.99**，比两件套 **3:13.41** 快 28.42s，整体约 **1.17×**；采样约 **1.21×**。
- 新进程 seed 44 真冷三件套为 **9:38.95**，采样仍 2:15。冷总时间与两件套几乎相同，因为 3:29–3:57 模型加载和约 3 分钟 VAE/文件缓存波动远大于 28 秒 attention 收益；不能据冷总时间否定采样提速。
- 生产 Torch 2.13 的旧两件套热基线 3:15 与本轮同环境 3:13.41 接近，数据相互印证。

### 13.8 三件套质量

- seed 42/43/44 三条 MP4 均为 124 帧、864×480、24 FPS、5.167s，H.264 + AAC 32kHz 双声道；帧唯一色、运动差和媒体流均健康。
- 联系表 `comfyui_download/sage3_sol_easy_comparison.jpg` 未见狗的 anatomy 崩坏、静帧、断帧或场景塌缩。
- 同环境三件套对两件套：seed 42/43 的视频 MAD 为 19.63/19.52，平均 PSNR 18.47/18.06dB，帧相关 0.827/0.821；说明 Sage3 在 SolAttn 只负责 dense 段时，轨迹变化小于纯 Sage3 对 SDPA（MAD 29.65），但仍不是无损。
- 音频 seed 42 相关 0.970、SNR 12.22dB；seed 43 相关仅 0.261，表明不同 seed 下音频轨迹可能明显分叉。当前只能确认结构健康，不能凭三条样本宣称质量等价。

### 13.9 更新结论

- **速度已确认**：Sage3 + SolAttn + EasyCache 的稳定热总耗时约 **2:45**，是本机当前最快的完整 H3 配置；相对现有两件套约快 28 秒。
- **稳定性初步通过**：合成烟雾、真实 4 步、三次 20 步完整生成均未出现 CUDA/native abort 或 GPU lost。
- **仍不直接替换生产**：该 wheel 锁定 Python 3.12/Torch 2.10，且视频/音频轨迹有明显数值变化。建议先扩展到多个 prompt/seed 的人工画面与听感回归；通过后再考虑把隔离环境升级为快速生成档。

---

## 十四、云端 ComfyUI 切换三件套与长时任务估算（2026-08-25）

### 14.1 历史耗时日志复核

- 仓库中可核验的最长相关实测是两段顺序工作流：总耗时 `16:14 = 974s`，净输出 226 帧，即 24 FPS 下 9.42s；第一段采样 2:55、第二段 3:18。证据在 `comfyui_download/run_s0102.log`。
- 当前 8188 的 `/history?max_items=200` 返回 0 条记录；仓库也没有“20s=49min”或“30s>2h”的完成日志。因此这两个数字只能作为用户提供的历史时间做比例估算，不能标记为项目实测。
- 三件套同环境热总耗时比两件套降低 14.7%（比例 0.8531）；采样段降低 17.2%（比例 0.8282）。长时任务包含多段初始化、Motion Context、解码和保存，推荐采用更保守的**完整总耗时比例 0.8531**。

| 用户历史时间 | 保守估算（×0.8531） | 预计节省 | 采样占主导上限（×0.8282） |
|---|---:|---:|---:|
| 8s 视频：1000s | **约 853s（14:13）** | 147s（2:27） | 约 828s（13:48） |
| 20s 视频：49min | **约 41:48** | 7:12 | 约 40:35 |
| 30s 视频：2h（按 7200s 下限） | **约 1:42:22** | 至少 17:38 | 约 1:39:23 |

30s 原描述是“2 小时多”，所以上表只用 2:00:00 做下限；实际优化后时间应把真实原始秒数乘以 0.8531。

### 14.2 云端服务切换

- `F:\python\llamacpp\deploy-cloud\start-comfyui-cloud.bat` 已改为启动 `ComfyUI_sage3_py312/venv`，设置 `COMFY_SAGE3=1`，继续监听本机 `127.0.0.1:8188`。
- 新 8188 已验证：Python 3.12、Torch 2.10.0+cu130、日志显示 `Using SageAttention3 Blackwell`；`SolAttnPatch`、内置 `EasyCache`、`H3DeepSeekPrompt`、`ResolutionSelector` 均已注册，NVFP4 TE 可见。
- 登录计划任务 `ComfyUI Cloud Tunnel` 已重新接管 watchdog，SSH PID 12600；云端 `127.0.0.1:8188` 返回 HTTP 200。
- 验证期间队列一直为空、history 为 0；**没有运行任何新生成任务**。

### 14.3 优化 prompt 与可见 UI 工作流

- API prompt：`comfyui_download/cloud_h3_sage3_solattn_easycache_prompt.json`。
- 云 UI 工作流：`comfyui_download/workflows/H3_DeepSeek_T2V_Sage3_SolAttn_EasyCache.json`，同时安装在 `ComfyUI_sage3_py312/user/default/workflows/`。
- 可见 MODEL 链为 `UNET（标题标记 Sage3 全局 backend） → SolAttnPatch → EasyCache → BasicGuider`；SolAttn 参数 tau 1.3 / INT8 QK / exact_kv_and_rows / Morton 2d_frame，EasyCache 0.30 / 0.2–0.9。
- 顶层新增醒目的 Markdown 说明，明确 Sage3 是启动级全局 backend、不会伪造一个无效 Sage 节点；原 DeepSeek 提示词、时长、分辨率和 seed 控件保留。
- 优化工作流默认使用 Qwen3VL NVFP4 AWQ 文本编码器，不再使用旧 INT8 ConvRot TE。

---

## 十五、Sage3 长任务崩 GPU：铁证 + "稳定"结论更正（2026-08-25）

### 15.1 触发与结论

在 8188（Python 3.12 / Torch 2.10.0+cu130 / ComfyUI 0.30.0）的 Sage3+SolAttn+EasyCache 三件套上，做 10s（`length=240` → 模型 snap 到 243 帧 / q=30509 tokens）全流程重渲染，**第 19/20 步把 GPU 硬崩**。这是继 [[sageattn2-crashes-h3-gpu]] 之后，本机又一次"Sage3/注意力内核原生 abort → GPU lost"事件。**Sage3 不是机械稳定。**

对用户此前"驱动没了我调整 10s（TDR）还是掉"的描述，本铁证给出明确答案：**不是 TDR 超时，是 CUDA 未知错误的原生崩溃**，所以把 `TdrDelay` 调到 10s 根本救不了。

### 15.2 本机幂等证据（同栈，两轮对照）

| 轮次 | prompt | seed | 结局 |
|---|---|---|---|
| 第 1 版（初版蜘蛛侠，skill-api 出剧本） | 无 | — | **成功** `Prompt executed in 00:14:41`，产物 `Cloud_H3_Sage3_SolAttn_EasyCache_00001_.mp4`（10.125s / 864×480 / H.264+AAC 32kHz / 帧 std 47–69） |
| 第 2 版（蜘蛛侠 vs 变形金刚，自写英文剧本直接注入） | 20260825 | — | **崩溃** 采样 `19/20` 时 `[ERROR] Error running SageAttention3: CUDA error: unknown error` → 原生 `abort()` → `nvidia-smi` 报 "GPU is lost. Reboot the system to recover this GPU" → 进程死、8188 无响应 |

崩溃栈（`cloud_server_error.log`）：`[ERROR] Error running SageAttention3: CUDA error: unknown error`，随后 torch `c10_cuda.dll` `warn_or_error_on_sync` → `VCRUNTIME140 _CxxThrowException` → `abort()`；`nvidia-smi` 复得 "No devices were found"。

### 15.3 关键推论

- **间歇性、内容/seed 相关**：同栈、同 240 帧，第 1 版跑完 14:41，第 2 版崩在末尾。所以"某一次 20 步/短帧跑到绿"**不能**证明长任务安全。
- **Sage3 主 attention 是崩点**：报错点名 `SageAttention3`，发生在采样阶段（非解码），序列 30509（比 124 帧的 15441 大一倍）。与 Sage2 那种"前向兼容 sm89 内核"不同，Sage3 是 Blackwell"原生"内核，但仍在此 GPU/长序列上触发 `CUDA error: unknown error`。
- **TDR ≠ 解药**：`CUDA error: unknown error` 是设备级原生错误，Windows TDR 超时机制拦不住（TDR 只处理可恢复的 <超时> 停顿），与 `TdrDelay=10` 无关。

### 15.4 配置决定

生产路径保持 **NVFP4 TE + SolAttn + EasyCache**（SDPA fallback，无 Sage3）。Sage3 三件套仅在**短/易恢复**任务上可用，且要预期可能 GPU reset；不建议作为长片默认。

工作流/产物：本次使用的注入式 10s prompt 在 `comfyui_download/`（`cloud_h3_sage3_solattn_easycache_prompt.json` 为基座，第 2 版自写剧本 `h3_spiderman_tf_giant_10s_prompt.json`）。

### 15.5 Kernel 级根因与"能不能修"分析（子代理扫源码 `models/SageAttention-for-windows`）

为判断"能否真正修好这个 wheel"，用只读子代理对放在 `models/SageAttention-for-windows/` 的官方 `thu-ml/SageAttention` 仓库源码做了深入扫描（详见下文，均为源码级推断、非运行时实锤）。

**崩掉的俩内核，源码都在仓库里：**

- `fp4attn_cuda.fwd` ← `sageattention3_blackwell/sageattn3/blackwell/api.cu`（fwd@338、mha_fwd@203、Blackwell gate@219）
- `fp4quant_cuda` ← `sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu`（pybind@624）

**关键结论一：不是"架构不支持"——是内核有真 bug。** `api.cu:219` 运行时仅检查 `major==12 && minor∈{0,1}`，RTX 5060 Ti（12,0）**正确通过**；编译侧 `setup.py` 只接受 `sm_100/120/121`，`sm_120a` 有生成。所以崩点不在"选错架构"。

**关键结论二：最可疑根因——per-block-mean 路径对 `delta_s` 越界读。** `api.py:preprocess_qkv` 把 L 从 30509 pad 到 **30592**，`qm` 只有 `ceil(L/128)=239` 行，但内核按 `m_block*128` 去索引 `delta_s`（假设它有 seqlen_q=30592 行；见 `api.cu:164-169` 设 `seqlen_s=seqlen_q`，`mainloop_tma_ws.h:476-482` 用 `m_block*128` 取 tile）。读得越深越易跨出缓存池撞到守卫页 → **间歇性 `CUDA error: unknown error`**，与"同 30509 序列一次成功一次崩"吻合。（探员明确标注为静态推断，未运行时确认，且质疑 `num_groups` 与 `seqlen_s` 的维度数学关系。）

**关键结论三：这个 wheel 我们修不了——无 Windows 重编通路。** `sageattention3_blackwell/setup.py` 是 Linux-only：只认 GCC 的 `-O3 -std=c++17 -lineinfo`、build 时 `git clone CUTLASS`、在 GPU 上取 `device_capability`、链 `libraries=["cuda"]`，**没有 MSVC 分支**。win_amd64 wheel 是外部环境编出来的。要在 Windows 重编 = 把 nvcc 旗标全面移植到 MSVC + 钉住 CUTLASS + 对齐 torch2.10/cuda13/py312，工程量大、风险高。
- 版本错位也印证：wheel 是 `cp312 + torch2.10.0 + cu130`，但 `sageattention3_blackwell/README.md:26` 声明 `python>=3.13 / torch>=2.8 / CUDA>=12.8`。

**关键结论四：官方"备选 fallback"也是雷。** 仓库提供非 FP4 路径 `sageattention.sageattn`（Sage2++ INT8-QK + FP8-PV），`core.py:152-153` 对 sm120 路由到 `sageattn_qk_int8_pv_fp8_cuda`——恰是之前 **Sage2 在第 3 步硬崩 GPU 的同一族**（`_qattn_sm89` 前向兼容到 sm120）。所以"换 Sage2++"只是换一个崩点。

**关键结论五：无任何环境变量能避开 FP4 路径。** 唯一参数 `per_block_mean`（api.py:131，默认 True）探员判定"未测试、不能可靠避雷"。

**底线：** Sage3 崩的原因不是架构误会，而是闭源预编译内核的一个长度相关越界 + 无 Windows 重编通路 + 备选又是同一崩家族 → **在我们这里没法实用地修复**。可靠解仍是 **SolAttn + EasyCache（SDPA fallback，无 Sage3）**；Sage3 只适合短任务/能接受再重启的场景。

相关：`sageattn3-blackwell-h3-stable` 记忆、`models/SageAttention-for-windows/`（源码）、`models/sageattn3-1.0.0%2Bcu130torch2.10.0-cp312-cp312-win_amd64.whl`（wheel）。

---

## 十六、Sage3 Windows 重编前法医复核（2026-08-25）

本节是第 15.5 节之后的新证据与迁移检查点。当前尚未修改 CUDA 源码、尚未重编 wheel，也没有启动任何 Sage3 GPU kernel；结论严格区分源码快照、已安装二进制和待验证假说。

### 16.1 源码与 PR #323 的精确来源

- 本地源码仓库：`models/SageAttention-for-windows/`，remote 为 `sdbds/SageAttention-for-windows`，当前 commit `5aa9dde3ef43a8f547fc02b3ae76e24ba6ccea6c`；它与上游的 merge-base 是 `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`。
- 已从 `thu-ml/SageAttention` 只读获取 PR #323 到本地 ref `refs/remotes/pr/323`；PR 当前 head 为 `142c02a3dbc3caa8d20eeb9cbad598f8c5a2429a`。运行期修复的原始 commit 是 `8bb81e4af5161c40ed96eaa6bb52f45c961ba6fc`。
- 本地 `kernel_ws.h` 与 `launch.h` 确实仍是四个大参数按值传递；整个 `sageattn3/blackwell` 目录没有 `_MSC_VER`，也没有 `cudaMallocAsync`/`cudaMemcpyAsync`/`cudaFreeAsync` 参数打包路径。
- 本地 `kernel_traits.h` 虽有 “Inline the definitions to avoid MSVC dependent-name quirks” 块，但它是无条件展开、写死 `BlockScaledConfig<16>` 的早期/局部写法，不是 PR #323 最终的 `_MSC_VER/#else` 版本。因此只能说源码树缺少 PR #323 的核心运行期补丁，不能仅凭这一点推断现有 wheel 的 ABI。

### 16.2 关键反证：现有运行 wheel 已经是指针 ABI

- 旧 wheel `models/sageattn3-1.0.0%2Bcu130torch2.10.0-cp312-cp312-win_amd64.whl` 的 SHA-256 为 `8728EE04CE619579F26B6B5AB84ED7B768CA8F44DA9ED2CB62EC85EBE40E3337`；wheel 元数据版本为 `1.0.0+cu130torch2.10.0`，tag 为 `cp312-cp312-win_amd64`。
- wheel 内 `fp4attn_cuda.cp312-win_amd64.pyd` 的 SHA-256 为 `F9AD8BEA2E2591517C501AA92A5B670471A18EB94A126B9DE7D4CCCF31AF6208`，与隔离 venv 中已安装的 `.pyd` 字节完全一致。
- 该 `.pyd` 中所有 `compute_attn_ws` CUDA kernel 符号的参数均为 `PK16Flash_fwd_params`、`PK...Mainloop...Params`、`PK...Epilogue...Params`、`PK...Scheduler...Params`；`PK` 是 Itanium ABI 的 pointer-to-const 编码。也就是说，当前实际运行的旧 wheel 已经采用四指针 kernel wrapper，不是源码快照里的四对象按值 ABI。
- 因此，“本地源码缺 PR #323”成立，但“当前长序列崩溃一定由旧 wheel 缺 PR #323 引起”不成立。重建仍有价值：它会让源码和制品可追溯，并加入 DS 64 位 stride；但不能预先承诺它一定修复 30.5k token 的间歇性 GPU lost。

### 16.3 delta_s / 对照实验的两处修正

- `mainloop_tma_ws.h` 的 `LayoutDS` 确实在 `make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0))` 中使用三个 32 位动态 stride。计划仅把这三个 stride 类型改为 `int64_t`，不改 32 位 shape，也不误改 `kernel_traits.h` 中相似但属于 `LayoutSF` 的代码。这对应 issue #382 在约 74k token 的 descriptor stride 溢出；30.5k token 的 batch stride 尚未超过 2³¹，所以它是前瞻修复，不是当前崩溃的直接证据。
- 本地 `api.cu` 中 `CHECK_SHAPE(delta_s, ...)` 已被注释；旧文档所称“会直接拒绝其他形状”不适用于这份源码。
- `per_block_mean=False` 也不是完全关闭 DS：它把 `seqlen_s` 设为 128、固定读取第 0 行，但仍创建 `TMA_DS`、执行 DS TMA copy，并无条件执行 `add_delta_s`。True/False 仍是有价值的二分，但 False 稳定不能等价解释为“DS 路径完全未参与”。
- 当前 `mainloop_params` 内有 7 个 TMA load 对象，`epilogue_params` 另有 1 个 TMA store 对象，合计 8 个，不是 7 个。

### 16.4 已锁定的重编环境与回滚

- 目标环境：Windows x64、RTX 5060 Ti sm_120、驱动 591.86；`ComfyUI_sage3_py312/venv` 为 CPython 3.12.0、Torch 2.10.0+cu130。
- 编译器：Visual Studio 2022 Build Tools `cl.exe 19.44.35222`；CUDA Toolkit 必须显式使用 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` 的 nvcc 13.0.48。当前普通 PATH 先命中 CUDA 12.8，不能直接裸跑 `nvcc` 构建。
- 旧 wheel 在 `models/` 和 `ComfyUI_sage3_py312/` 各有一份相同 SHA-256 的备份，作为 golden rollback。新 wheel 必须输出到独立、版本化目录，不能覆盖旧制品。
- 当前 8188/8191 无监听，也没有隔离 venv 的 Python 进程；但登录计划任务可能再次拉起该环境，安装或测试前必须复查。

### 16.5 下一检查点（尚未执行）

1. 成对移植 PR #323 的 `kernel_ws.h` + `launch.h`，并给 `setup.py` 加入 PR 的 MSVC/CUDA 13 编译分支；保留非 MSVC 原 ABI。
2. 在 `mainloop_tma_ws.h` 仅做 `LayoutDS` 三个动态 stride 的 `int32_t → int64_t` 修改。
3. 固定 CUTLASS 来源，使用 VS2022 x64 developer environment + nvcc 13.0 构建到新目录，记录 source/PR/CUTLASS commit、完整命令、wheel SHA-256 和成员哈希。
4. 先做 wheel CRC/metadata/tag、离线 staging import 和短序列 1152/4096 验证；再逐级到 15441。`q=30509` 已有 GPU lost 实证，只能在明确接受 WDDM reset、已保存工作且可立即重启时人工执行，不纳入普通自动验证。

### 16.6 重建执行记录（2026-08-25）

§16.5 第 1、2 步（PR #323 移植 + `mainloop_tma_ws.h` DS stride int64）**已落在工作树**：`git status` 显示 `kernel_ws.h / launch.h / mainloop_tma_ws.h / setup.py` 均 M，`csrc/`（CUTLASS）已 clone。`mainloop_tma_ws.h` 的 +1/-1 确认为：

```diff
- make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))
+ make_stride(int64_t(0), _1{}, int64_t(0), int64_t(0)))
```

（对应 issue #382 的 descriptor stride 溢出；30.5k token 的 batch stride 尚未爆 2³¹，属前瞻修复，非当前崩溃的直接证据。）

**构建环境实测**：VS2022 BuildTools（`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`）`cl.exe 19.44.35222`；CUDA v13.0 `nvcc 13.0.48`（PATH 默认先命中 CUDA 12.8，**必须显式 `CUDA_HOME=v13.0` + 前置其 `bin`**）；隔离 venv CPython 3.12.0 / Torch 2.10.0+cu130；CUTLASS 已 clone 于 `sageattention3_blackwell/csrc/cutlass`。

**首次构建失败（已修）**：torch`cpp_extension` 在 VC 环境要求 `DISTUTILS_USE_SDK=1`（否则报 "multiple activations of the VC env"）。已在构建脚本 `models/SageAttention-for-windows/build_sage3_ds64.bat`（vcvars64 之后）加 `set DISTUTILS_USE_SDK=1` + `set MSSdk=1`，后续构建进入 `building 'fp4attn_cuda' extension` 正常阶段。

构建脚本：`models/SageAttention-for-windows/build_sage3_ds64.bat`；日志 `.../build_sage3_ds64.log`。

### 16.7 构建纠偏：DS int64 改动回退，只带 PR #323（2026-08-25）

**复查 PR #323 的权威 diff**：`git diff d1a57a54..142c02a3 -- sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h` 为**空** —— PR #323 **根本不改 `mainloop_tma_ws.h`**。因此：

- §16.5/16.6 里"mainloop DS stride int64"那处 **是我方添加的 #382 前瞻改动**，**不属于 PR #323**。（子代理也标注：它只防 74k-token 的 descriptor stride 溢出，30.5k 未爆 2³¹，**不是当前崩溃的直接证据**。）
- 该改动**编译失败**：把 `LayoutDS`（line 55-62）stride 改 int64 后，line 224 `tile_to_shape(...)`（int32 形状/动态维度）与之不匹配；改形状为 int64 又造成形状类型不匹配。根因是 Cutlass `Layout` 的 shape/stride 类型需一致，单纯改 stride 不成立。
- **处置**：`git checkout -- mainloop_tma_ws.h` 回退到 HEAD（line 60 恢复 `make_stride(int32_t(0),...)`）。当前仅保留 **PR #323 的三处**：`kernel_ws.h`、`launch.h`、`setup.py`（MSVC 分支）。重建改为 **PR #323-only**，版本串 `1.0.0+cu130torch2.10.0.msvcpr323`。

**教训**：不要在同一仓库里叠加"自以为的前瞻修复"到已验证的 PR 之上；先以**上游权威 diff** 为准，确认某改动确属修复范围再做。DS 溢出的 #382 是独立问题，与本次 30.5k 崩溃无关。

### 16.8 重建 + 分阶段验证结果（2026-08-25）

**构建**：`build_sage3_ds64.bat`（vcvars64 + `DISTUTILS_USE_SDK=1` + `CUDA_HOME=v13.0`）→ `EXITCODE=0`，产出 `build/lib.win-amd64-cpython-312/fp4attn_cuda.cp312-win_amd64.pyd` / `fp4quant_cuda...pyd`，MSVC 19.44.35222 / nvcc 13.0.48 / torch 2.10.0+cu130。
- 新 fp4attn_cuda SHA-256 `d53df7397c502ffa880f9854b8601e4c922cd31fad7129fd5ce083f837cf855c`（golden `f9ad8bea...`），fp4quant `86110881...`（golden `e9056d6a...`）。golden 备份于 `models/rebuild_sage3_golden_backup_231836/`。
- ⚠️ §16.2 已指出：golden wheel 的 fp4attn_cuda **本来就已是 PR #323 的指针 ABI**。所以"重建是否真的改变了内核逻辑"存疑——下方单次 30509 通过不能排除 golden 只是"恰好那次内容/时序没崩"。

**分阶段验证（全部通过）**：
| 阶段 | 结果 |
|---|---|
| 离线 import（venv 换入新 .pyd 后 `import fp4attn_cuda/fp4quant_cuda`） | ✅ |
| 短序列合成（1,24,1152/4096,128 & 1,56,15441,128） | ✅ 输出有限、形状正确；15441 约 109ms |
| **30509 场景合成（1,56,30509,128）** | ✅ **输出有限、形状正确、约 392ms、未崩 GPU**（这是 golden wheel 崩溃的确切形状） |

**当前判读（保守）**：重建内核通过了单次 30509 合成调用，**中止了 kernel 级崩溃**；但原始崩溃发生在完整渲染 `19/20`（内容/seed/累积状态相关），单次调用**不能**完全排除完整场景仍崩。下一步才是真正的"针对场景"——完整 14 分钟渲染验证；一旦 GPU lost 立即回退 golden（`models/rebuild_sage3_golden_backup_231836/`）。

### 16.9 完整场景验证：PR #323 重建**未能**修复崩溃（决定性负结果，2026-08-25）

对重建 PR #323 内核做**完整 14 分钟场景渲染**（8188，`COMFY_SAGE3=1`，用重建内核；workflow `h3_spiderman_tf_giant_10s_prompt.json`，10s / length=240 / q≈30344）：

- 单次合成 30509 通过≠完整场景稳定。**完整渲染崩溃**：`Fatal Python error: Aborted`，发生在 **`comfy/samplers.py:1333 in sample`（第 2/20 步）**；日志中**没有** `Error running SageAttention3` 行，说明 fatal abort 发生在 Sage3 内核调用内部、`try/except` 未及捕获。
- **这次崩在第 2/20 步，比 golden 崩的第 19/20 更早、更接近 init** → 进一步佐证崩溃**内容/累积状态相关、间歇性**，与 §16 的 delta_s 越界/Blackwell 内核推测一致。
- **结论**：PR #323（kernel_ws/launch 四指针 ABI）**不是本例崩溃的修复**。加上 §16.2（golden 已含 PR #323 ABI）→ 重建≈golden 逻辑，故必然仍崩。**Sage3 在 sm120 上跑 H3 长序列本质上不稳定**，不是 wheel 版本/ABI 能修——它更可能是内核源码级逻辑 bug（delta_s 越界或 Blackwell 特定问题）。

**处置**：venv 已回退 golden `.pyd`（SHA `f9ad8bea`/`e9056d6a` 恢复）；`models/rebuild_sage3_golden_backup_231836/` 保留备份。**决策不变**：生产 = NVFP4 TE + SolAttn + EasyCache（SDPA fallback，无 Sage3）。Sage3（或 Sage2）对 H3 长序列**不可用**，若要用只能短任务且接受再次 GPU lost/重启。

### 16.10 外部复核修正：PR #323 本就不是修这类崩溃（2026-08-25，用户提供的上游一手分析）

**核心修正（推翻 §16.6–16.8 的"缺参数打包"判断）：**

- **PR #323（mengqin fork）修的是另一类崩溃**：woct0rdho#42 / #357 —— Windows sm_120 上**确定性、小形状（`(4,32,64,128)`）必现**的 `misaligned address`。根因：源码能编但缺 `/Zc:__cplusplus` → CUTLASS 宏未启用 → C++17 特性禁用 → 数据对齐问题；CUDA 13.0 下 `CUTE_GRID_CONSTANT` 正确启用后 kernel 参数要 128 字节对齐、与 MSVC 仅 16 字节对齐冲突 → 只能改指针/引用传参。**与我的"间歇、30k、数分钟、GPU lost"无关。**
- **"缺参数打包"假说已证伪**（§16.2/16.8 ABI 检查）：golden 与重建两个 wheel 的 kernel 符号都是 `PK16Flash_fwd_params…`（引用/指针 ABI），**都崩** → 崩溃与参数对齐/打包无关。我上一轮基于"按值传参快照"的推断**不成立**。
- **快照 commit 澄清**：
  - 快照 HEAD = `5aa9dde change to C++20 for torch2.13`，此处 `kernel_ws.h` 是**按值** `CUTE_GRID_CONSTANT ... const params`、`launch.h` **无** `_MSC_VER` —— 即 pre-PR 按值版。
  - 工作树（本次构建用）`kernel_ws.h` 为 `..._impl(const & params)`（引用 ABI）、`launch.h` 含 `#if defined(_MSC_VER)` —— 即 **PR #323 参考 ABI 版**（未提交改动）。
  - golden wheel 二进制 = `PK16...`（引用 ABI）≈ 工作树。故 golden 与重建**同一套 ABI**，我未拿错树；§16.9"重建≈golden 逻辑故必崩"成立。

**崩溃指纹对不上任何已修/已开 issue：**

| 问题 | 错误 | 复现 | 平台 | 状态 |
|---|---|---|---|---|
| woct0rdho#42 / #357 | misaligned address | 确定性、小形状 | Win sm_120 | **已被 PR #323/fork 修复** |
| #382 | host 端 TMA descriptor 初始化失败→illegal instruction | 确定性、~74k token | Linux sm_120 | open，无修复 |
| **本例** | **unknown error → GPU lost** | **间歇，15k–30.5k，数分钟后** | **Win sm_120 (WDDM)** | **无对应 issue** |

30.5k 时 batch stride ≈4.1×10⁸ 元素，离 2³¹ 还远，不踩 #382（74k）的坑；**加的 DS int64 是正确但超前的预防**（§16.7 已回退）。上游 sm_120（36 SM vs 测试的 5090）占用/调度不同，且 #392 有 CUDA graph replay 静默错误 —— "上游 Linux 测试全过"≠ 这条路径无潜伏 race。

**剩余候选（按嫌疑排序，均能解释"间歇+长序列+数分钟后"）：**
1. **内核 race**：warp-specialized producer/consumer + 自定义 `OrderedSequenceBarrier`（"Group 0 反向相位"很脆），36 SM 上占用/调度与测试的 5090 不同；负载越大越易触发 —— race 典型指纹。
2. **TDR 级联**：atomic kernel ~392ms（远低于 10s TdrDelay），但若 barrier 等待死锁 → kernel 挂死 → TDR 杀设备 → `unknown error` → `GPU lost`；解释了为何错误在 sync 处冒尖。
3. **显存瞬时峰值**：30.5k×56 头 fp32 delta_s 每调用现分配 ~1.64GB + packed QKV/SF/O(~438MB)，叠在 H3 采样 13GB+ 基线；WDDM 分配/分页失败可表现为 device lost（#391 native 对照削弱纯 VRAM，但 Sage3 峰值更高）。
4. **构建变量未锁定**：`setup.py` 每次 build 现 `git clone --depth 1` CUTLASS master，golden 与新 wheel 可能用了不同 CUTLASS 快照 → "重编验证"非受控实验。

**定位动作（待重启后执行）：**
1. 脱离 ComfyUI 独立 repro：`[1,56,30509,128]` bf16 循环 100 次，`per_block_mean` True/False 分两个子进程（崩溃会毒化 CUDA context）。False 也崩 → 与 DS 路径无关；False 稳定 → DS TMA 路径实锤。
2. 同 repro 下 `compute-sanitizer --tool memcheck`（Windows 可用）区分 OOB/race/descriptor；`CUDA_LAUNCH_BLOCKING=1` 定位出错 launch。
3. 钉 CUTLASS commit 重编一次，排除构建变量。
4. 拿 repro + sanitizer 日志去 #382 追加评论或开新 issue（30.5k 阈值低于 74k 且 Windows，是有价值的新数据点）。生产继续用已验证的 SolAttn/EasyCache，Sage3 等上游修复。

### 16.11 重建内核时间戳确认 + 复测进行中（2026-08-25/26）

**时间戳/provenance 核对**（确认正在跑的 `cd119308` 用的是刚重编的 PR #323 内核，非旧版）：
- 重建编译完成：`build/lib.win-amd64-cpython-312/fp4attn_cuda.cp312-win_amd64.pyd` mtime = **2026-08-25 23:16:26**（本次会话 MSVC/nvcc13 构建，SHA `d53df73…`）。
- 装进 venv site-packages：mtime = **2026-08-26 00:27:56**（起 Sage3 前 `cp -f`）。
- 8188 进程（netstat PID 9048）启动晚于 00:27:56 → 加载的是刚装的重建内核。
- venv 当前 SHA = `d53df73…`（重建版，非 golden `f9ad8bea…`）。

**复测进度（`cd119308`，Sage3 + 重建内核，10s / q=30344）**：init 3:50 → 采样 1/20 @00:50、2/20 @00:50、**3/20 @01:15**（25.27s/it），GPU 100% / 79°C。已**过步 2**（此前重建测试崩于第 2 步）。继续监控至完成或崩；若完成即"重建内核在完整场景成立"（与 PR#323 非本 crash 类的判读相悖，需再查），若崩则再次确认 PR #323 无效。

### 16.12 定案：Sage3 放弃，生产定 = SolAttn + EasyCache（2026-08-26）

**本轮完整场景复测 `cd119308`（Sage3 + 重建 PR #323 内核）最终仍崩**（过程中一度过步 2、进入采样深处，但未跑完）。GPU 曾 lost 需重启——**至此累计 3 次重启**，成本明确，止损。

**结论（最终）：**
- Sage3（连同 Sage2）在 RTX 5060 Ti / sm_120 上跑 H3 长序列（15k–30.5k token）**本质不可用**；崩溃为**间歇、内容/时序相关**（小形状/单次合成过、完整渲染崩），符合**内核 race / 潜伏 bug**，无上游修复，且经 PR #323（不同 crash 类）、MSVC 重编、ABI 核对均**无法修复**。
- **cu128 假设评估后放弃**：崩溃是 race（非编译器版本类），编译器差异通常造成确定性区别，难以解释随机崩溃；且 venv torch 为 cu130，用 nvcc 12.8 编内核有 cudart 12.8/13.0 ABI 冲突风险 + `sm_120a` 在 nvcc 12.8 是否支持未实测。试一次=再开 20 分钟编译+大概率第 4 次重启，成本/收益不成立。

**生产定案 = NVFP4 TE + SolAttn + EasyCache（SDPA fallback，无 Sage3）。** 注意：`start-comfyui-cloud.bat` 目前设 `COMFY_SAGE3=1`（Sage3 全局 backend），若要跑生产稳定栈，**须改为 `COMFY_SAGE3=0`（或去掉该行）**，否则 8188 仍会用 Sage3 并崩。

**遗留（可选，供未来上游跟进，非生产必需）**：`comfyui_download/sage3_repro.py`（per_block_mean True/False 独立 repro，未跑）；CUTLASS 未钉版本；`models/SageAttention-for-windows/` 源码与 PR #323 改动、`models/rebuild_sage3_golden_backup_231836/` 备份、venv 已回退 golden `.pyd`（SHA `f9ad8bea`/`e9056d6a`）。


