# 研究报告:MiniMax H3 在 ComfyUI 输出乱码的问题根源

**日期**: 2026-08-03
**研究方式**: Perplexity 研究(pro 模式,1 次主搜索 + 1 次 follow-up)+ 本地实证

---

## 关键发现(按置信度排序)

### 1. ComfyUI 的 H3 支持仍处于 Day-0/早期阶段,输出乱码很可能是上游实现问题(置信度:中)
- MiniMax H3 的 ComfyUI 原生节点是 v0.30.0 刚加入的(Day-0 支持),社区有输出不稳定/异常的讨论。
- 社区对比显示 **SGLang 是更成熟的 H3 部署路径**,ComfyUI 路径尚在逐步完善。
- 本机实证: 工作流连线与官方模板完全一致、模型反量化验证正确(MAE<0.01),但输出仍乱码 → 指向 ComfyUI 实现本身的问题,而非用户配置错误。

### 2. comfy_quant 空字节是已知的加载适配问题(置信度:中高)
- Comfy-Org/MiniMax-H3 的 pruned int8 权重 `comfy_quant` 张量为空,已知有社区补丁/分支尝试修复加载逻辑。
- 本机处理: 通过原位改写 comfy_quant 为正确 JSON(fl2va: int8_tensorwise+convrot256; text_encoder: nvfp4),使模型能加载且反量化经 bf16 真值验证正确。

### 3. qwen3vl nvfp4 文本编码器加载有已知坑(置信度:中高)
- 版本对齐与加载顺序是常见问题点。
- 本机实证: text_encoder 条件输出对不同 prompt 有**固定的巨大离群值(max=15974, std 完全相同)**,虽反量化逐层验证正确,但深层存在放大问题。

### 4. 单卡 16GB 显存跑 H3 需要低显存策略(置信度:中)
- 社区方案: --lowvram、注意力优化(xformers/SDP)、tiled VAE、先低分辨率再放大。
- 本机验证: 16GB + 64GB 可运行(峰值显存 14.5GB/16GB),5 秒 480p 视频约 7-9 分钟。

---

## 项目文档审查

- `F:\python\h3\docs.md`(本次会话操作记录,RELIABLE —— 本会话亲历):
  - 模型文件全部下载并验证(fl2va/text_encoder 反量化 vs bf16 MAE 均 <0.01)。
  - 官方模板连线 = 手工工作流。
  - 所有生成视频(含官方模板)均乱码,已删除。
  - 下载过程反复出现"满大小但区域清零"的损坏。

---

## 矛盾与缺口

- Perplexity 回答无具体 URL(占位 [1][2]),无法直接核证"已知修复提交"。
- 官方模板使用 v0.4 subgraph 格式(UUID 复合节点),当前 ComfyUI master 需前端展开,本环境浏览器 MCP 不可用,未能在 UI 中直接跑官方模板(改为 API 等效构建,结果一致乱码)。
- text_encoder 离群值的**确切来源**(哪个层/哪个量化细节)未定位。

---

## 建议(后续动作)

1. **等待上游修复**: ComfyUI H3 原生节点或 Comfy-Org 权重更新,关注 GitHub issues。
2. **改用 SGLang**: H3 官方推荐路径,更成熟,但需 Linux + 多卡(本机 Windows 单卡不适用)。
3. **改用 MiniMax 云端 API**: 绕开本地量化问题,立即可用(工具包有现成脚本)。
4. **若继续 ComfyUI 排错**: 优先定位 text_encoder 离群值来源(逐层检查中间 hidden states),或尝试 GGUF 量化版。

---

## 搜索覆盖

- Perplexity pro: 1 次主搜索(5 项清单)+ 1 次 follow-up(5 项聚焦)。
- 本地实证: 模型反量化 bf16 对比、官方模板子图解析、下载损坏扫描与修复、多次生成测试。
- 未覆盖: ComfyUI GitHub issue 逐条核实(需代理,Perplexity 未给具体 issue 号)。
