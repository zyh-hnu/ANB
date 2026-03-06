# Centralized SAFB

该目录提供集中式 SAFB 训练入口，用于验证“联邦聚合是否导致攻击效果退化”。

## 核心设计

- 复用当前项目 SAFB 触发器实现（`core/attacks.py`），不改攻击核心。
- 训练策略参考 `FIBA-main/train.py` 的混合思路：
  - `poison`：注入触发器并改为目标标签（提升 ASR）
  - `cross`：注入触发器但保持原标签（稳住 ACC）
  - `clean`：原始样本
- 输出与联邦实验口径一致的指标：`ACC`、`ASR`（可选 `ASR_multi`）。

## 快速运行

在项目根目录执行：

```bash
python centralized/train_safb.py --dataset CIFAR10 --freq-strategy ANB --epochs 50 --poison-rate 0.2 --cross-ratio 1.0 --target-label 0 --epsilon 0.1
```

## 常用参数建议

- 如果想先追求高 ASR：提高 `--poison-rate`（例如 `0.3`），降低 `--cross-ratio`。
- 如果想保持更高 ACC：降低 `--poison-rate`（例如 `0.1~0.2`），提高 `--cross-ratio`。
- 若你要复现实验中的 ANB 全功能，保持以下开关为 1：
  - `--use-phased-chaos 1`
  - `--use-spectral-smoothing 1`
  - `--use-freq-sharding 1`
  - `--use-dual-routing 1`

## 结果文件

默认输出目录：`results/centralized_runs/<run_name>/`

- `history.json`：逐 epoch 曲线（`test_acc`, `test_asr` 等）
- `summary.json`：最终与最优指标摘要
- `config.json`：完整参数快照
- `model_final.pth`：最后一轮模型
- `model_best_asr.pth`：ASR 最优模型
- `model_best_tradeoff.pth`：满足最小 ASR 约束时 ACC 最优模型（若存在）
