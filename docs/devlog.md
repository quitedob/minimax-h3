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
2. **SolAttn 仍是 H3 唯一稳定的采样加速**(attention ↓55%,20步 ↓26%)。
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
