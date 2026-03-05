# ANB 毕业论文代码优化计划

> **加载说明**: 每次对话开始时请先阅读此文件，以增量方式追加新内容，不要覆盖历史记录。
> **最后更新**: 2026-03-02

---

## 一、项目现状总结

### 1.1 研究定位

**系统名称**: SAFB (Semantic-Aware Frequency Backdoor) / ANB (Adaptive Nebula Backdoor)

**核心主张**: 提出一种针对联邦学习的自适应频域后门攻击，能够绕过 FreqFed（NDSS 2024）基于 DCT 聚类的防御机制。

**攻击四大特性**:
1. **Phased Dynamic Chaos** — 三阶段相位调度（稳定→扩张→混沌）
2. **Normalized Spectral Smoothing** — 高斯扩散的"星云"频率触发器
3. **Frequency Sharding** — 以客户端ID为索引的频段分片，打散聚类特征
4. **Dual-Domain Routing** — 基于局部方差的频域/空域双路由注入

**基线对比**: FIXED策略（所有客户端使用同一频段，即FIBA-like基线）

### 1.2 代码架构评分（当前状态）

| 模块 | 完整度 | 学术严谨性 | 问题 |
|------|--------|------------|------|
| `core/attacks.py` | ★★★★☆ | ★★★☆☆ | 缺乏理论推导支撑，phase 参数硬编码 |
| `core/defenses.py` | ★★★★☆ | ★★★★☆ | 仅复现FreqFed，无增强防御对比 |
| `federated/server.py` | ★★★☆☆ | ★★★☆☆ | 缺少模型替换缩放，FedAvg 聚合权重未做 clip |
| `federated/client.py` | ★★★☆☆ | ★★☆☆☆ | 无 scaling factor，恶意客户端影响力严重不足 |
| `data/distribution.py` | ★★★★☆ | ★★★★☆ | Dirichlet 正确实现，但未统计各客户端分布 |
| `models/resnet.py` | ★★★★★ | ★★★★★ | CIFAR-10 适配正确 |
| `main.py` | ★★★☆☆ | ★★☆☆☆ | 实验对比组不完整，缺消融实验入口 |
| 评价指标 | ★★☆☆☆ | ★★☆☆☆ | 缺 LPIPS、SSIM、BA、CDA 等关键指标 |

---

## 二、从论文角度的核心问题诊断

### 2.1 【CRITICAL】恶意客户端影响力不足（最高优先级）

**位置**: `federated/client.py` `Client.train()`

**问题**: 2个恶意客户端在10个客户端中仅占20%权重，FedAvg后其后门梯度被8个良性客户端大幅稀释，导致 ASR 难以突破 50%。

**标准做法**: 学术界通行做法是对恶意客户端施加 **Model Replacement Scaling**（模型替换缩放）:
```
scaled_update = (global_weights + delta * λ)
其中 λ = num_clients / num_malicious = 10/2 = 5.0
```

**修复方案**: 在 `Client.train()` 返回前，对恶意客户端的权重差做缩放：
```python
# 在 client.py 的 train() 末尾
if self.is_malicious:
    scaling_factor = 1.0 / (len_malicious / num_clients)  # e.g., 5.0
    global_w = global_model.state_dict()
    scaled = {}
    for k in local_weights:
        scaled[k] = global_w[k] + (local_weights[k] - global_w[k]) * scaling_factor
    return scaled, self.num_samples, avg_train_loss
```

**论文影响**: 这是 ASR 能否达到论文声称目标（>85%）的关键前提，**必须修复**。

---

### 2.2 【CRITICAL】测试 ASR 计算包含 target 类样本（评价指标错误）

**位置**: `data/dataset.py` `PoisonedTestDataset.__getitem__()`

**问题**: 第171行 `self.backdoor(image_np, original_label)` 中，当 `original_label == target_label` 时，`AdaptiveNebulaBackdoor.__call__` 会直接返回原图（不毒化）。但 `PoisonedTestDataset` 仍把这些样本的标签设为 `target_label`，并计入 ASR 分母——实际上这些样本本身就是目标类，模型原本就应预测为 target，会虚高 ASR。

**正确做法**: 在 `PoisonedTestDataset.__len__` 和 `__getitem__` 中，只使用 `original_label != target_label` 的样本。

**修复方案**:
```python
# PoisonedTestDataset.__init__ 中过滤掉目标类
self.valid_indices = [
    i for i in range(len(base_dataset))
    if base_dataset[i][1] != target_label
]
def __len__(self): return len(self.valid_indices)
def __getitem__(self, idx):
    real_idx = self.valid_indices[idx]
    ...
```

---

### 2.3 【HIGH】评价指标不完整，论文说服力不足

**现有指标**: ASR, ACC (在 `server.py` 中)

**缺失的关键指标**（硕士论文必须包含）:

| 指标 | 含义 | 当前状态 | 优先级 |
|------|------|----------|--------|
| **BA (Benign Accuracy)** | 无攻击时的基线准确率 | 缺失 | 必须 |
| **CDA (Clean Data Accuracy)** | 有后门时在干净数据上的准确率，等同当前ACC | 已有但命名不规范 | 低 |
| **PSNR** | 触发器不可见性（峰值信噪比，越高越好，>30dB） | 分析脚本有，主实验无 | 必须 |
| **SSIM** | 结构相似性，补充PSNR | 缺失 | 推荐 |
| **LPIPS** | 感知距离，用于衡量人眼不可见性 | 依赖已安装但未调用 | 推荐 |
| **Defense Recall/Precision/F1** | 防御绕过的定量评估 | 分析脚本有，主实验history无 | 必须 |
| **L∞ Norm of Perturbation** | 触发器扰动上界，对应 epsilon 约束 | 缺失 | 推荐 |

**需要新增** `analysis/metrics.py` 模块统一计算上述指标，并在每轮训练后记录。

---

### 2.4 【HIGH】消融实验入口缺失

**问题**: 论文实验部分通常需要消融实验验证各组件的贡献，目前代码没有提供关闭单个特性的开关。

**需要增加的消融配置**:

```python
# config.py 需新增
use_phased_chaos: bool = True      # 消融：去掉阶段性相位调度
use_spectral_smoothing: bool = True # 消融：去掉高斯扩散，改为点频触发
use_freq_sharding: bool = True      # 消融：去掉分片，所有客户端用同一频段
use_dual_routing: bool = True       # 消融：去掉空域路由，仅用频域
```

**对应论文表格（示例）**:

| 配置 | ASR (with defense) | ACC |
|------|--------------------|-----|
| Full ANB | XX% | XX% |
| w/o Phased Chaos | XX% | XX% |
| w/o Spectral Smoothing | XX% | XX% |
| w/o Freq Sharding | XX% | XX% |
| w/o Dual Routing | XX% | XX% |
| FIXED Baseline | XX% | XX% |

---

### 2.5 【MEDIUM】训练曲线记录不完整

**位置**: `federated/server.py` `print_round_summary()`

**问题**: `history['defense_results']` 未记录 Recall/Precision/F1，`history['test_asr']` 在防御开启时测量的是绕过防御后的 ASR，但没有区分"防御过滤前"和"过滤后"的实际聚合结果。

**需要增加**:
```python
self.history['defense_recall'] = []
self.history['defense_precision'] = []
self.history['psnr_per_round'] = []
self.history['accepted_malicious_ratio'] = []  # 被防御漏过的恶意客户端比例
```

---

## 三、创新点强化方向

### 3.1 现有创新点评估

| 创新点 | 新颖性 | 可信度 | 论文能否支撑 |
|--------|--------|--------|-------------|
| Frequency Sharding | ★★★★☆ | ★★★☆☆ | 需要理论分析支撑（为何分片能骗过DCT聚类）|
| Phased Dynamic Chaos | ★★★☆☆ | ★★☆☆☆ | 参数经验性设置，缺乏理论依据 |
| Normalized Spectral Smoothing | ★★★☆☆ | ★★★☆☆ | 能量补偿公式 `1 + sigma*1.5` 是经验值，需要推导 |
| Dual-Domain Routing | ★★★★☆ | ★★★☆☆ | GradCAM实验可验证其有效性 |

### 3.2 【推荐】可强化的理论贡献

**A. 为 Frequency Sharding 提供理论分析**

FreqFed 的核心假设是：恶意客户端在权重的 DCT 特征空间中会形成异常聚簇。

ANB 的反制理论可以形式化为：
- 设 $\mathbf{f}_i \in \mathbb{R}^d$ 为客户端 $i$ 的 DCT 特征向量
- FreqFed 需要 $\text{cosine\_distance}(\mathbf{f}_{m_1}, \mathbf{f}_{m_2}) \ll \text{cosine\_distance}(\mathbf{f}_{m}, \mathbf{f}_{b})$（恶意客户端相互相似，与良性客户端相异）
- ANB 通过 Sharding 使 $\text{cosine\_distance}(\mathbf{f}_{m_1}, \mathbf{f}_{m_2}) \approx \text{cosine\_distance}(\mathbf{f}_{m}, \mathbf{f}_{b})$

**在图表中应展示**: 防御特征空间 t-SNE 可视化（已有 `visualize_clusters.py`），需配合真实训练权重。

**B. 增加对更多防御的测试**

目前只对比 FreqFed。可以增加：
- **FLTrust** (CCS 2022): 基于根数据集的信任评分聚合
- **Foolsgold**: 基于梯度相似性的惩罚机制
- **FLAME** (USENIX 2022): 基于噪声注入的鲁棒聚合

哪怕只是简单复现并展示 ANB 对它们的影响，也会大幅提升论文贡献度。

---

## 四、代码工程质量问题

### 4.1 轻微但需要修复的 Bug

**Bug 1**: `server.py:130` — FedAvg 使用 `template_weights = client_weights_list[0]` 作为模板，若 `accepted_clients` 不包含客户端0，会导致 key casting 逻辑基于错误模板。
```python
# 修复：改为使用接受列表中的第一个
template_weights = client_weights_list[accepted_clients[0]]
```

**Bug 2**: `federated_training()` 的早停机制 (`test_asr > 0.95`) 只在无防御时才触发，但没有防止在防御开启时因某轮防御失效而 ASR 误触发早停的情况。正确逻辑是无防御时早停即可，当前代码已处理，但应加日志说明。

**Bug 3**: `data/distribution.py:32` — `counts` 为 int 数组，`counts[-1]` 调整可能产生负数（当某类样本极少时）。需要 `counts[-1] = max(0, ...)` 保护。

### 4.2 性能优化

**问题**: `AdaptiveNebulaBackdoor._generate_normalized_nebula_pattern()` 在每次图像采样时都重新生成完整 grid，在大 batch 下性能较差。

**建议**: 预计算 `grid_x, grid_y` 并缓存（对固定图像尺寸有效）：
```python
# __init__ 中预计算
self._cached_grid = {}  # (H, W) -> (grid_x, grid_y)
```

### 4.3 可复现性问题

**问题**: `_get_current_phase()` 在 Stage 2/3 使用 `np.random.randint`，每次调用结果不同，即使设置了全局 seed，因为 `set_round()` 内部不重置 RNG 状态。

**影响**: 相同 round，相同 client，不同 epoch 的 batch 可能使用不同 phase，导致训练/测试时触发器不一致，使 ASR 测量不稳定。

**建议**: 为随机 phase 引入确定性种子：
```python
rng = np.random.RandomState(seed=self.client_id * 1000 + self.current_round)
idx = rng.randint(0, 4)
```

---

## 五、实验设计完善计划

### 5.1 必须完成的实验对比（论文核心表格）

**Table 1: Main Results**

| Method | Defense | ASR (↑) | ACC (↓降不多) | Defense Recall (↓) |
|--------|---------|---------|----------|----------------|
| BadNets | None | ~99% | ~90% | - |
| FIBA-like (FIXED) | None | XX% | XX% | - |
| **ANB (Ours)** | None | XX% | XX% | - |
| BadNets | FreqFed | ~30% | ~90% | 100% |
| FIBA-like (FIXED) | FreqFed | XX% | XX% | XX% |
| **ANB (Ours)** | FreqFed | XX% | XX% | XX% |

### 5.2 消融实验表格

见 §3.2 消融配置。

### 5.3 参数敏感性分析（超参消融）

| 参数 | 测试范围 | 固定其他值 |
|------|---------|----------|
| epsilon | {0.05, 0.1, 0.15, 0.2} | 其他默认 |
| poison_ratio | {0.1, 0.2, 0.3, 0.4} | 其他默认 |
| alpha (Non-IID) | {0.1, 0.3, 0.5, 1.0} | 其他默认 |
| num_rounds | {30, 50, 100} | 其他默认 |

### 5.4 跨数据集泛化实验

- CIFAR-10（已有）
- CIFAR-100（代码已支持，需运行）
- 可选：GTSRB（交通标志，更接近真实场景）

---

## 六、论文章节对应的代码支撑清单

| 论文章节 | 需要的图/表 | 对应代码/脚本 | 状态 |
|---------|-----------|------------|------|
| Introduction | 动机图（防御失效示意） | `create_defense_evasion_illustration()` | ✅已有 |
| Method: Attack | 触发器生成流水线图 | `visualize_trigger_generation_pipeline()` | ✅已有 |
| Method: Frequency Sharding | 多客户端频段分布图 | `visualize_multi_client_triggers()` | ✅已有 |
| Method: Dual-Domain Routing | 方差图 + 路由mask可视化 | ❌ 缺失 | 需新增 |
| Experiments: Main Table | ASR/ACC数值 | 主实验运行 | ⚠️ 需修复ASR计算 |
| Experiments: Defense | t-SNE聚类图（FIXED vs ANB）| `visualize_cluster_results()` | ⚠️ 需真实权重 |
| Experiments: Imperceptibility | PSNR/SSIM/LPIPS表格 | `evaluate_imperceptibility.py` | ⚠️ 缺SSIM |
| Experiments: Ablation | 消融实验表 | ❌ 缺失 | 需新增config开关 |
| Experiments: Sensitivity | 参数敏感性曲线 | ❌ 缺失 | 需新增脚本 |
| Appendix: GradCAM | GradCAM对比图（安全区域） | `gradcam_check.py` | ⚠️ 需验证 |

---

## 七、增量执行计划（按优先级排序）

### Phase 1：紧急修复（影响实验结果正确性）

- [x] **P1-1**: 修复 `PoisonedTestDataset` 过滤目标类样本（ASR计算正确性）
- [x] **P1-2**: 添加恶意客户端 scaling factor（ASR能否达标的前提）
- [x] **P1-3**: 修复 `server.py:130` FedAvg 模板选择 bug

### Phase 2：实验完整性（论文核心数据）

- [x] **P2-1**: 在 `config.py` 增加消融开关（4个特性的 on/off）
- [x] **P2-2**: 在 `server.py` 的 history 中增加 defense_recall/precision/F1
- [x] **P2-3**: 新增 `analysis/metrics.py`，统一计算 PSNR/SSIM/LPIPS
- [x] **P2-4**: 修复 phase 随机性问题（引入 client_id + round 确定性种子）
- [ ] **P2-5**: 运行主实验（无防御 / FIXED+防御 / ANB+防御）并记录数据

### Phase 3：创新点强化（提升论文深度）

- [x] **P3-1**: 新增 Dual-Domain Routing 可视化（方差mask图）
- [x] **P3-2**: 添加至少1个额外防御基线（建议 FLTrust 或 Foolsgold）
- [x] **P3-3**: 参数敏感性分析脚本
- [x] **P3-4**: CIFAR-100 实验

### Phase 4：论文写作支撑

- [x] **P4-1**: 整理所有图表、数据到统一结果文件夹，格式化为论文可用的高分辨率图
- [ ] **P4-2**: 编写完整的 README 实验复现指南

---

## 八、关键代码修改记录（增量追加）

> 每次实际修改代码后，在此处追加记录。

| 日期 | 文件 | 修改内容 | 状态 |
|------|------|---------|------|
| 2026-03-02 | plan.md | 初始创建，完成全面代码审阅 | ✅ |
| 2026-03-02 | data/dataset.py | P1-1: PoisonedTestDataset/MultiTriggerTestDataset 过滤目标类，修正ASR计算 | ✅ |
| 2026-03-02 | config.py | P1-2+P2-1: 添加 scaling_factor 及4个消融开关 | ✅ |
| 2026-03-02 | federated/client.py | P1-2: 恶意客户端 Model Replacement Scaling | ✅ |
| 2026-03-02 | federated/server.py | P1-3+P2-2+P3-2: FedAvg模板修复、history扩展、统一防御接口 | ✅ |
| 2026-03-02 | main.py | P1-2+P2-1: scaling_factor传递、build_backdoor_factory消融开关 | ✅ |
| 2026-03-02 | core/attacks.py | P2-1+P2-4: 4个消融开关注入、确定性RNG种子 | ✅ |
| 2026-03-02 | core/defenses.py | P3-2: 新增 FLTrust、Foolsgold、aggregate_with_defense 统一接口 | ✅ |
| 2026-03-02 | analysis/metrics.py | P2-3: 新建，统一 PSNR/SSIM/LPIPS/L∞ 不可见性评估 | ✅ |
| 2026-03-02 | analysis/visualize_dual_routing.py | P3-1: 新建，双域路由机制可视化 | ✅ |
| 2026-03-02 | analysis/sensitivity.py | P3-3: 新建，参数敏感性分析脚本 | ✅ |
| 2026-03-02 | analysis/cifar100_experiment.py | P3-4: 新建，CIFAR-100 跨数据集泛化实验 | ✅ |
| 2026-03-02 | analysis/collect_figures.py | P4-1: 新建，图表归集整理脚本，输出 results/paper/ | ✅ |

---

## 九、参考对标论文（用于定位贡献）

1. **FreqFed** (NDSS 2024) — 被攻击的防御，已有PDF
2. **FIBA** (CVPR 2022) — 频域后门基线
3. **BadNets** (IEEE S&P 2019) — 空域后门基线，最简单的基线
4. **FLTrust** (NDSS 2022) — 替代防御对比候选
5. **DBA** (ICLR 2020) — 分布式后门攻击，概念上与本工作相关（每个客户端注入不同片段）

> DBA 是最重要的相关工作：本工作的 Frequency Sharding 与 DBA 的分布式触发器思想相似，**必须在论文 Related Work 中明确区分**：DBA 在空域分片（像素块），ANB 在频域分片（频段），且 ANB 额外具有动态性和自适应性。

---

*此文件为增量更新，后续每次工作后追加到"八、关键代码修改记录"和对应 Phase 的完成状态中。*

