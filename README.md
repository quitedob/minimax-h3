# MiniMax-H3 本地部署

MiniMax-H3 视频生成模型在本机（Windows 11 / NVIDIA RTX 5060 Ti 16GB / 64GB RAM / CUDA 12.8）的本地部署记录与相关脚本。

## 硬件环境

| 组件 | 规格 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| 内存 | 64 GB |
| 系统 | Windows 11 / CUDA 12.8 / Python 3.12 |
| 推理框架 | ComfyUI（便携版） |

## 内容

- `docs.md` — 完整的本地部署全过程记录（模型下载、量化转换、显存策略、实测指标）
- `research-h3-garbage.md` — MiniMax H3 在 ComfyUI 输出乱码问题的研究记录
- `models/` — 模型文件（ModelScope / HuggingFace 下载）
- `ComfyUI_windows_portable/` — ComfyUI 便携版运行环境

## 相关链接

- 模型来源：ModelScope / HuggingFace MiniMax-H3
- 推理路径：ComfyUI（原生 H3 节点，Day-0 支持）或 SGLang（更成熟路径）
