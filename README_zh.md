# 在一颗 AWS Trainium2 芯片上 30 分钟训练语言模型

这是一道基于 [AWS Trainium Frontier](https://trainium-frontier.devpost.com/) 比赛第一阶段的测试题，在比赛结束后按比赛规则完成。任务是在一颗 Trn2 芯片上从零训练 GPT 风格的语言模型，限时 30 分钟，目标是在固定评测上得到尽可能低的 val_bpb（每字节比特数）。

**结果：val_bpb 1.0605，starter kit 基线为 1.2591（降低 0.199）。** 四天内共排队运行了 130 个实验。

## 主要工作

- **发现一个静默的精度 bug**：一个形状为 `[1]` 的 fp32 标量乘到 bf16 残差流上，PyTorch 的类型提升让之后所有矩阵乘都以 fp32 运行。用 forward hook 打印每个 Linear 的输入 dtype 定位到问题。把 Linear 输入转成 bf16 后每步从 1.53 秒降到 0.78 秒，同配置 val_bpb −0.023。
- **NKI kernel**：融合 RoPE + QK-RMSNorm 的前向和反向。利用旋转不改变向量长度，每 128 行一个 tile 只读写一次 HBM，前向快 37%，步数多 7%。CPU 和设备上都对照 fp32 参照验证过。
- **训练循环提速**：单卡 batch 8、torch.compile 编译模型以及函数式的 Muon/AdamW、学习率改成设备上的 0 维张量（Neuron eager 模式会把 Python 浮点数编进 kernel，每步重编译，从 1.5 秒变 8 秒）。加上 bf16 修复，每秒 token 从 41K 提到 168K，同时模型从深度 10 加到 12。
- **把速度换成规模**：深度 14（1.496 亿参数）、每步 384K token，由完整 30 分钟的 batch、深度、Muon 学习率扫描确定。
- **实验系统化**：实例上的 runner 从 git 分支拉取实验队列，逐个运行、评测并把日志和结果推回 GitHub，无需保持 SSH 连接，芯片全天不停。每次运行都是“某个提交 + 环境变量覆盖”，可以从结果表复现。
- **发现并修复评测陷阱**：影响前向但没存进 checkpoint 的设置在评测时会用默认值。现在 logit softcap 以及 bf16 matmul、fused QKV、两个 NKI kernel 开关都存在模型配置里。

## 目录

| 路径 | 内容 |
| --- | --- |
| `train.py` | 最终训练脚本 |
| `baseline/train.py` | starter kit 原版，方便对比 |
| `prepare.py`、`requirements.txt`、`artifacts/` | 未修改的 starter kit：数据、tokenizer、评测、提交检查 |
| `REPORT.md` | 两页报告（英文） |
| `results.tsv` | 全部 130 次运行：提交、val_bpb、步数、秒/步、保留/丢弃/崩溃、说明 |
| `experiments/` | 实验说明、原始结果表、实验队列、runner 脚本、37 个关键运行的日志、图 |

实验在另一个未公开的仓库中完成，每个实验一个提交；`results.tsv` 和 `experiments/` 里的提交哈希指向那份历史。本仓库包含最终代码、完整结果表和 37 个关键运行的日志。

## 许可

基于 AWS Trainium Frontier 第一轮 starter kit（Apache 2.0）。`train.py` 在其基础上修改，见 `NOTICE` 与 `LICENSE`。代码、实验和分析借助 AI 编程助手（Claude）完成，它也负责驱动实验队列。
