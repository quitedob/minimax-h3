# Windows H3 / Sol-Attn 源码核对（2026-09-11）

## 范围与结论

这是源码与导入兼容性检查，不是安装或性能验收。遵循用户的原生 Windows 限制；没有安装 WSL、升级 ComfyUI/依赖、修改生产工作流、重启服务或 SSH、提交视频生成。

- **高置信度**：Saganaki22 主分支 HEAD 为 `930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf`，对应 `v0.6.2`，提交时间 `2026-08-13T16:17:38+01:00`；本次读取的远端标签没有更新版本。因此“该仓库主分支 9 月没有更新”成立，但不据此推断整个生态没有更新。
- **高置信度**：该实现有原生 Windows / SM120 Triton pointer 路径，不要求安装 WSL。README 的开发基准为 ComfyUI 0.30.0，测试版本为 0.30.1，不能说必须 >=0.32.0。[上游说明](https://github.com/Saganaki22/ComfyUI-sol-attn/tree/930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf)
- **高置信度**：现有 `ComfyUI-SolAttn_triton` 已经使用无 contiguous 复制的 INT8 pointer 内核，并有 Morton 排序，不能把 Saganaki22 的“零拷贝”宣传当作本机新增收益。
- **中置信度，静态推断**：Saganaki22 的 Fused Modulation 仍调用模块级 AdaLN、attention、MLP；未发现绕过本机 Turbo AdaLN 注入的直接路径。FFN 分块调用原 MLP forward，也未直接绕过 LoRA。但完整模型、量化、流式换入及组合稳定性尚未验收。
- **未验证**：Saganaki22 相对本机当前 8 步栈的真实速度、峰值显存、视频/音频质量。没有新增基准结果。

## 本地环境及已有基线

当前解释器：`F:\python\h3\ComfyUI_sage3_py312\venv\Scripts\python.exe`。

预检：Windows 11、Python 3.12.0、Torch 2.10.0+cu130、Triton-Windows 3.6.0.post26、comfy-kitchen 0.2.26；RTX 5060 Ti，SM120，15.93 GiB VRAM，驱动 591.86。RAM 63.84 GiB、检查时可用 33.53 GiB；F 盘可用 77.15 GiB。GPU 与 Torch CUDA 检查均通过。PATH 中 nvcc 为 CUDA 12.8；这不等于 Torch 所用 CUDA 版本。

生产图仍为：pruned FP8 H3 基座 + NVFP4 文本编码器 + 双 VAE；Turbo v4 EMA、strength 1.0、simple、8 steps、low_vram=false；SolAttn INT8 QK、tau 1.3、0.2–0.9、exact_kv_and_rows、Morton 2d_frame；EasyCache 0.30、0.2–0.9；SDPA fallback，不启用 Sage2/3。

`docs/devlog.md:832`、`:843` 的已有测试为 864×480、124 帧、24 FPS，Turbo 热生成 149.618 秒。它与历史 20 步结果的比较不是同日多轮严格 A/B；本次没有重测，也不能把它当成新插件的结果。

## 源码证据与组合差异

候选源码目录：`comfyui_download/ComfyUI-sol-attn_Saganaki22/`，不在生产 `custom_nodes/` 内，clone 后工作树干净。

| 检查项 | 源码证据 | 对当前栈的影响 |
|---|---|---|
| SM120 dispatch | 候选 `sol_kernel/fwd.py:106` 的 `_use_pointer_arch` | 返回 pointer；没有 Windows 强制 dense 门槛 |
| 现有零拷贝 | 生产 `custom_nodes/ComfyUI-SolAttn_triton/__init__.py:112` | 已把交错 QKV 的 strided views 传给内核 |
| 现有 Morton | 生产同目录 `_morton_h3.py:124` | 当前图显式启用；候选代码未提供同名/对应排序选项 |
| 候选稀疏调度 | `minimax.py:443`、`:724` | 用模型 sigma 计算 tau 和 early dense gate，不是按“8 步中的第几步”线性计数；参数不能机械照搬旧 start/end |
| 候选 attention | `minimax.py:338`、`:396` | 替换 H3 attention.forward，仍调用 qkv_proj/out_proj；不能将多个接管同一路径的 Sol 节点随意堆叠 |
| 调制融合 | `minimax.py:169`、`:194`、`:227` | 保留 AdaLN、attention、MLP 模块调用；未知既有 block.forward 补丁会跳过 |
| Turbo AdaLN | 生产 `custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py:378` | DIFFUSION_MODEL wrapper + adaln_proj.forward 注入，与候选融合的模块调用在静态路径上可衔接 |
| FFN 分块 | 候选 `minimax.py:35`、`:105` | 分段调用原 MLP forward，需验证 FP8 + bypass LoRA 的数值及性能 |

候选在独立进程、现有解释器中完成导入检查，结果：

```text
registered_nodes = MiniMaxH3ChunkFeedForward,
                   MiniMaxH3FusedModulation,
                   MiniMaxH3MemoryEfficientSolAttentionPatch,
                   MiniMaxH3ScheduledSolAttentionPatch,
                   SolAttentionPatch
sol_kernel_importable = True
sm120_uses_pointer = True
kernel_executed = False
model_loaded = False
production_modified = False
```

## 对用户提供材料的修正

1. 1.73–1.97× 是上游 RTX 5090 注意力微基准相对 Sage 的吞吐比，不是相对本机旧 Sol 的整段视频加速。37% 是 INT8 ConvRot 检查点的 MLP 激活峰值下降，不是总显存下降；本机是 FP8 基座。[基准说明](https://github.com/Saganaki22/ComfyUI-sol-attn/blob/930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf/README.md#benchmarks)
2. 官方 Sparse Attention PR #16072 实际在 **9 月 6 日**合入，不能沿用“尚未合入”的旧路线说明。[官方 PR](https://github.com/Comfy-Org/ComfyUI/pull/16072)
3. 当前 comfy-kitchen 主分支能力矩阵已经包含 CUDA `sol_attn`，并说明提供 Windows CUDA wheels。因此“CUDA Sol-Attn 整体仍仅存在于 PR、必需自己编译”的表述已经过时；某个额外优化 PR 是否进入特定 wheel，仍需按版本核对，本次未安装或执行新版内核。[官方仓库](https://github.com/Comfy-Org/comfy-kitchen)
4. NVlabs/Sana 的 `models/minimax_h3/Sol-H3` 与这些 ComfyUI 节点不是同一运行时。此前已在 `comfyui_download/Sana_sol_engine/` 阅读 `8249b285142e70ab596370db59636a714de3169b`：该 H3 engine 的非 dense 路径要求多 GPU，加载方式也不是当前 FP8/NVFP4 动态换入栈；vendored interface 在缺少 CuTe 时存在 Triton 分派。不能笼统称所有 Sol-H3 在 Windows 上一律 dense。[指定上游](https://github.com/NVlabs/Sana/tree/8249b285142e70ab596370db59636a714de3169b/models/minimax_h3/Sol-H3)

## 建议的后续验证（尚未执行）

先保留原注意力和 8 步图，单独 A/B 调制融合，再 A/B FFN 分块；两项均可在不切换注意力后端时评价。之后另设候选图比较 Saganaki22 residual INT8，不同时启用两套 Sol，也不重新启用历史失败的 Sage2/3。

比较固定模型、LoRA、提示词、seed、864×480、124 帧、24 FPS、8 步及缓存参数。Triton 首次编译单列；热启动重复测量采样、解码、总时间与资源峰值，确认实际内核和 Turbo 注入，再检查多帧与音轨。Morton 与 sigma 调度差异要明确记录，不能宣称逐项等价替换。ComfyUI 核心升级应另立隔离对照，不与上述补丁一次混改。

## 项目文档可靠性与覆盖范围

- `docs/devlog.md` §18：**RELIABLE（在所记录范围内）**；已交叉检查生产模板的模型名、8 步、LoRA 和补丁配置。历史耗时仍只是历史实验。
- §16.12：**PARTIALLY RELIABLE**；多次崩溃和不启用 Sage 的决定可作本机操作依据，但“本质不可用 / race”等根因表述不能当成已证明的上游结论。
- 已有只读文档 scout 的结果用于定位；本次主代理重新检查关键 devlog、配置及源码。
- 检查日期为 2026-09-11；范围为用户给定仓库、已有项目文件和先前定位的一手 PR。Perplexity 认证未恢复，未使用其他搜索服务代替发现新来源。没有逐条验证社区文章、工作流热度、全部生态时间线、AMD 分支或 5060 Ti 以外的性能。
- 本次新增内容只有阅读用 clone 和本报告；没有生产变更需要回退。
