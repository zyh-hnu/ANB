# SAFB 集中学习最佳实验配置

## 实验结果对比分析

### 已运行实验汇总 (5次实验)

| Run Name | Epochs | Cross Ratio | Spectral | Dual Routing | Final ACC | Final ASR | Best ASR | Best ASR ACC | Tradeoff ACC | Tradeoff ASR |
|----------|--------|-------------|----------|--------------|-----------|-----------|----------|--------------|--------------|--------------|
| `073054` | 8 | 0.5 | 0 | 0 | 21.55% | 76.16% | 91.67% | 13.1% | 13.1% | 91.67% |
| `073325` | **50** | **0.5** | **0** | **0** | **64.95%** | **67.56%** | 91.67% | 13.1% | 13.1% | 91.67% |
| `074748` | 20 | 1.0 | 0 | 0 | 48.00% | 32.99% | **97.10%** | 8.6% | 8.6% | 97.10% |
| `080216` | 50 | 0.5 | 1 | 1 | 64.85% | 26.47% | 95.40% | 10.1% | 15.1% | 85.26% |
| `081757` | 20 | 0.5 | 0 | 1 | 44.60% | 36.11% | 95.40% | 10.1% | 15.1% | 85.26% |

### 关键发现

1. **最佳配置: Run `073325`**
   - Final ACC: 64.95%, Final ASR: 67.56%
   - 50 epochs + cross_ratio=0.5 + 关闭所有高级特性
   - 这是目前 ACC/ASR 最平衡的配置

2. **Epochs 影响**
   - 8 epochs: 训练不充分，ACC仅21.55%
   - 50 epochs: 最佳整体表现，ACC达64.95%
   - 20 epochs: 中等效果

3. **Cross Ratio 影响**
   - `cross_ratio=0.5`: 最佳ACC/ASR平衡
   - `cross_ratio=1.0`: Final ACC降至48%，Final ASR仅32.99%

4. **高级特性影响** (Spectral Smoothing & Dual Routing)
   - **关闭时 (073325)**: Final ASR 67.56%
   - **开启时 (080216)**: Final ASR 降至 26.47%
   - 结论: 这些特性在集中学习场景下反而降低 ASR，应保持关闭

5. **训练曲线特征**
   - 所有实验 Best ASR 都在 91%-97% 区间
   - 但 Best ASR 对应的 ACC 都很低 (8%-13%)
   - 说明需要找到 ACC-ASR 的最佳权衡点

---

## 最佳推荐配置

基于 5 次实验结果分析，推荐以下配置：

### 核心配置

```python
BEST_CONFIG = {
    # 数据集配置
    "dataset": "CIFAR10",
    "data_dir": "./data",
    "model_name": "resnet18",

    # 后门攻击配置
    "backdoor_name": "frequency",
    "freq_strategy": "ANB",          # 自适应频域策略
    "target_label": 0,
    "epsilon": 0.1,                  # 触发器强度

    # 投毒配置
    "poison_rate": 0.15,             # 投毒率
    "cross_ratio": 0.5,              # 跨域样本比例 (实验证明最佳)
    "backdoor_boost_weight": 0.2,    # 后门增强权重

    # 训练配置
    "epochs": 50,                    # 充分训练
    "batch_size": 256,               # 可根据GPU调整
    "learning_rate": 0.05,           # 初始学习率
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "milestones": "30,45",           # 学习率衰减节点
    "gamma": 0.1,

    # 数据配置
    "train_subset": 0,               # 0=使用全部数据
    "test_subset": 0,
    "num_workers": 2,

    # SAFB 特性开关 (集中学习场景推荐关闭)
    "use_phased_chaos": 1,           # 启用阶段混沌
    "use_spectral_smoothing": 0,     # 关闭 (实验证明降低ASR)
    "use_freq_sharding": 1,          # 启用频域分片
    "use_dual_routing": 0,           # 关闭 (实验证明降低ASR)

    # 评估配置
    "min_asr_for_tradeoff": 0.85,    # Tradeoff评估阈值

    # 其他
    "seed": 42,
    "device": "auto",
}
```

### 命令行启动

```bash
python centralized/train_safb.py \
    --dataset CIFAR10 \
    --freq-strategy ANB \
    --target-label 0 \
    --epsilon 0.1 \
    --poison-rate 0.15 \
    --cross-ratio 0.5 \
    --backdoor-boost-weight 0.2 \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 0.05 \
    --milestones "30,45" \
    --use-phased-chaos 1 \
    --use-spectral-smoothing 0 \
    --use-freq-sharding 1 \
    --use-dual-routing 0 \
    --output-dir results/centralized_runs \
    --run-name best_config_run
```

---

## 配置说明

### 1. 关键参数对比

| 参数 | 实验验证 | 说明 |
|------|----------|------|
| `epochs` | 50 | 最佳训练时长 |
| `cross_ratio` | 0.5 | 1.0 会导致性能下降 |
| `use_spectral_smoothing` | 0 | 开启后 ASR 从 67.56% 降至 26.47% |
| `use_dual_routing` | 0 | 开启后 ASR 下降 |

### 2. SAFB 特性开关

| 开关 | 推荐值 | 原因 |
|------|--------|------|
| `use_phased_chaos` | 1 | 阶段性调整触发器，提升多样性 |
| `use_spectral_smoothing` | 0 | 集中学习场景下降低 ASR |
| `use_freq_sharding` | 1 | 频域分片，增强触发器多样性 |
| `use_dual_routing` | 0 | 集中学习场景下降低 ASR |

### 3. 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 0.05 | 配合 Milestones 衰减 |
| `milestones` | [30,45] | 适配 50 epochs |
| `batch_size` | 256 | 可根据 GPU 调整 |

---

## 实验改进建议

### 待验证方向

1. **全量数据训练**
   - 当前使用 subset (8000 train / 2000 test)
   - 建议使用完整数据集验证

2. **学习率调整**
   - 尝试更小的初始学习率 (0.01)
   - 或使用 Cosine Annealing

3. **投毒率优化**
   - 当前 0.15，可尝试 0.1 或 0.2

4. **多客户端评估**
   - 设置 `--eval-multi-client-ids "1,2,3"`
   - 评估跨客户端触发器泛化能力

---

## 预期性能指标

基于 Run `073325` 实验结果：

| 指标 | 实际值 | 目标值 |
|------|--------|--------|
| Final ACC | 64.95% | > 65% |
| Final ASR | 67.56% | > 65% |
| Best ASR | 91.67% | > 90% |

---

## 文件引用

- 训练脚本: `centralized/train_safb.py`
- 实验配置: `kaggle/run_experiment.ipynb`
- FIBA参考: `FIBA-main/train.py`, `FIBA-main/config.py`
- 结果目录: `results/centralized_runs/`

---

*文档更新时间: 2026-03-06*
*基于5次集中学习实验结果分析*