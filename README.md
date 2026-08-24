# MiniMax-H3 本地部署

MiniMax-H3 视频生成工作区：ComfyUI 便携版、H3 工作流、SolAttn 加速和可选的 DeepSeek prompt 改写。

## 硬件环境

| 组件 | 规格 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| 内存 | 64 GB |
| 系统 | Windows 11 |
| 推理框架 | ComfyUI portable |
| 当前主环境 | Python 3.13 · Torch 2.13.0+cu130 |

## DeepSeek Prompt 节点

根目录 `.env` 配置 API key（`.env` 不入库）：

```dotenv
DEEPSEEK_API_KEY=你的_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

首次配置可复制 `.env.example` 为 `.env`，然后填入真实 key。不要把真实 key 写进工作流或提交到 Git。

启动 ComfyUI：

```text
ComfyUI_windows_portable/run_nvidia_gpu.bat
```

重启后，在 ComfyUI 的 **Load → Workflows** 中打开：

- `H3_DeepSeek_T2V.json`：纯文本生成视频
- `H3_DeepSeek_I2V.json`：首帧图片生成视频
- `H3_DeepSeek_R2V.json`：图片/视频/音频参考生成视频

工作流中的输入流程统一为：

```text
用户输入想法 → H3 DeepSeek Prompt → MiniMax H3 prompt → 采样 → 视频
```

用户只需要在 **H3 DeepSeek Prompt** 节点的 `user_text` 输入自然语言。节点会把 `.claude/skills/h3-prompt-writing/` 下的 `SKILL.md`、`references/base-en.txt` 和 `references/ref-en.txt` 一起发送给 DeepSeek，再把返回的完整 H3 prompt 传给 T2V/I2V/R2V 节点。

`mode` 要与工作流匹配：T2V 用 `T2VA`，I2V 用 `I2VA`；R2V 工作流默认使用 `Ref2VA`。I2V 的图片仍在工作流的 `LoadImage` 节点中选择，R2V 的参考素材按工作流中的参考输入连接。

## 目录

- `docs/` — 部署、性能和 attention 研究记录
- `models/` — 本地模型权重，不入库
- `ComfyUI_windows_portable/` — ComfyUI 便携版运行环境
- `comfyui_download/workflows/` — 可分发的 H3 DeepSeek 工作流
- `ComfyUI_windows_portable/ComfyUI/custom_nodes/h3_deepseek_prompt/` — DeepSeek 自定义节点
- `.claude/skills/h3-prompt-writing/` — H3 prompt 规范和模板

## 相关链接

- 模型来源：ModelScope / HuggingFace MiniMax-H3
- 推理路径：ComfyUI 原生 H3 节点 + SolAttn
