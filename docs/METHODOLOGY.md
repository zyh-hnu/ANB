# ANB 攻击方法详解

本文档详细介绍 ANB (Adaptive Nebula Backdoor) 攻击方法、基线对比以及 FreqFed 防御机制。

---

## 一、攻击方法：ANB (Adaptive Nebula Backdoor)

### 1.1 核心思想

ANB 是一种**自适应频率域后门攻击**，专门针对联邦学习场景下的 FreqFed 防御进行优化设计。

传统后门攻击在空域注入触发器，容易被防御检测。ANB 将触发器注入到频域，并采用多种自适应策略来规避聚类防御。

### 1.2 四个核心组件

| 组件 | 英文名称 | 功能 | 实现细节 |
|------|----------|------|----------|
| **动态相位调度** | Phased Chaos | 根据训练阶段调整触发器相位 | 早期稳定(0°)，后期随机化增加隐蔽性 |
| **高斯扩散** | Spectral Smoothing | 将点状触发器扩散为星云状 | σ 从 0.8 自适应增长到 1.5 |
| **频率分片** | Frequency Sharding | 不同恶意客户端使用不同频率位置 | 按客户端 ID 分配到预设的 "安全区域" |
| **双域路由** | Dual Routing | 根据图像内容自适应注入方式 | 纹理区域→频域，平坦区域→空域 |

### 1.3 触发器生成原理

```
原始图像
    ↓
频域变换 (FFT)
    ↓
在特定频率位置注入高斯扩散的正弦波
    ↓
频域逆变换 (IFFT)
    ↓
与原图融合 (ε 控制强度)
    ↓
带后门的图像
```

**关键代码** (`core/attacks.py`):

```python
# 1. 在频域生成星云模式
nebula = self._generate_normalized_nebula_pattern(H, W, center_u, center_v)

# 2. 计算双域路由掩码
freq_routing, spatial_routing = self._compute_dual_routing_masks(image)

# 3. 注入触发器
freq_inject = nebula_3d * epsilon * freq_routing_3d * 1.5  # 频域分支
spatial_inject = spatial_pat * spatial_routing_3d * epsilon * 0.6  # 空域分支

# 4. 融合
poisoned = img_float + freq_inject + spatial_inject
```

### 1.4 动态调度策略

**相位调度**：

| 训练阶段 | 轮次范围 | 相位选择策略 | 目的 |
|----------|----------|--------------|------|
| 稳定期 | 0-15 | 固定按客户端ID分配 (0°, 90°, 180°, 270°) | 稳定学习后门 |
| 扩展期 | 15-35 | 从主相位池随机选择 | 增加多样性 |
| 混沌期 | 35+ | 从全部8个相位随机选择 | 最大化隐蔽性 |

**扩散系数调度**：

```python
def _get_adaptive_sigma(self):
    if self.current_round < 20:
        return 0.8   # 早期：尖锐的星状触发器，信号强度高
    else:
        return 1.5   # 后期：模糊的星云状触发器，隐蔽性高
```

### 1.5 频率分片池

预设的频率 "安全区域"，避开 DC 分量和高频噪声：

```python
freq_shards = [
    (2, 2), (2, 3),     # Shard 0 → Client 0, 5, ...
    (3, 2), (3, 3),     # Shard 1 → Client 1, 6, ...
    (2, 5), (5, 2),     # Shard 2 → Client 2, 7, ...
    (3, 5), (5, 3),     # Shard 3
    (4, 5), (5, 4),     # Shard 4
    (4, 4)              # Fallback
]
```

---

## 二、基线对比：FIXED 策略

### 2.1 FIXED 是什么

FIXED 是 ANB 的简化版本，关闭所有自适应特性，作为实验基线。

### 2.2 FIXED 与 ANB 对比

| 特性 | FIXED (基线) | ANB (本文方法) |
|------|-------------|----------------|
| 相位策略 | 固定为 0° | 动态调度 (0°→90°→180°→270°→随机) |
| 扩散系数 | 固定 σ=0.8 | 自适应 σ=0.8→1.5 |
| 频率位置 | 所有客户端使用同一位置 | 按客户端 ID 分散到不同位置 |
| 双域路由 | 关闭 | 开启 |
| 抗聚类能力 | 弱 | 强 |

### 2.3 代码实现差异

```python
# FIXED 策略：所有自适应特性关闭
if self.strategy == 'FIXED':
    return 0.0                                    # 静态相位
    return 0.8                                    # 固定扩散系数
    center_u, center_v = self.freq_shards[0]      # 所有客户端同一频率位置
    return np.ones(...), np.zeros(...)            # 无双域路由

# ANB 策略：所有自适应特性开启
if self.strategy == 'ANB':
    return self._get_current_phase()              # 动态相位
    return self._get_adaptive_sigma()             # 自适应扩散
    center = self.freq_shards[self.client_id % N] # 分散频率位置
    return self._compute_dual_routing_masks(...)  # 双域路由
```

### 2.4 实验对比目的

| 对比维度 | 预期结果 |
|----------|----------|
| 无防御场景 | FIXED 和 ANB 都能达到高 ASR |
| FreqFed 防御下 | FIXED 被检测，ANB 绕过防御 |
| 隐蔽性 | ANB 的 PSNR/SSIM 更优 |

---

## 三、防御方法：FreqFed

### 3.1 FreqFed 是什么

FreqFed 是一种**基于频域聚类的联邦学习防御方法**，发表于 NDSS 2024。

**核心假设**：恶意客户端的模型更新在频域具有异常特征（因为注入了频域触发器）。

### 3.2 防御流程

```
客户端更新列表
    ↓
提取频域特征 (FFT → 能量分布)
    ↓
聚类分析 (HDBSCAN / KMeans)
    ↓
识别异常簇 (小簇或离群点)
    ↓
过滤恶意更新
    ↓
FedAvg 聚合
```

### 3.3 本项目实现

**核心代码** (`core/defenses.py`):

```python
def freqfed_defense(client_weights_list, client_num_samples, global_weights, 
                    malicious_indices=None, method='hdbscan'):
    
    # 1. 提取每个客户端更新的频域特征
    features = []
    for weights in client_weights_list:
        delta = compute_delta(weights, global_weights)  # 计算更新量
        freq_feature = extract_frequency_energy(delta)  # 频域能量分布
        features.append(freq_feature)
    
    features = np.array(features)
    
    # 2. 聚类分析
    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        labels = clusterer.fit_predict(features)
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(features)
    
    # 3. 识别异常簇 (通常是较小的簇或噪声点)
    accepted = identify_normal_cluster(labels)
    
    return accepted
```

### 3.4 聚类方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **HDBSCAN** | 自动确定簇数，识别噪声点 | 对参数敏感 | 默认方法 |
| **KMeans** | 简单快速 | 需预设簇数 | 二分类场景 |
| **DBSCAN** | 识别任意形状簇 | 对密度参数敏感 | 非均匀分布 |

### 3.5 评估指标

| 指标 | 含义 | 计算公式 | 理想值 |
|------|------|----------|--------|
| **Recall** | 恶意客户端被检测出的比例 | TP / (TP + FN) | 高 (防御方视角) |
| **Precision** | 被标记恶意中真正恶意的比例 | TP / (TP + FP) | 高 |
| **F1 Score** | Precision 和 Recall 的调和平均 | 2×P×R / (P+R) | 高 |
| **Bypass Rate** | 恶意客户端绕过防御的比例 | FN / (TP + FN) | 低 (防御方视角) / 高 (攻击方视角) |

**注意**：从攻击方视角，Bypass Rate 越高越好（攻击成功绕过防御）。

---

## 四、系统架构

### 4.1 联邦学习攻击场景

```
┌─────────────────────────────────────────────────────────────────┐
│                      联邦学习系统                                │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ 恶意客户端0  │  │ 恶意客户端1  │  │    良性客户端 2-9       │ │
│  │             │  │             │  │                         │ │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │  ┌─────────────────┐   │ │
│  │ │ ANB攻击 │ │  │ │ ANB攻击 │ │  │  │   正常本地训练   │   │ │
│  │ │ 频域触发│ │  │ │ 频域触发│ │  │  │   (无后门)      │   │ │
│  │ └─────────┘ │  │ └─────────┘ │  │  └─────────────────┘   │ │
│  │             │  │             │  │                         │ │
│  │ poison_rate │  │ poison_rate │  │     poison_rate = 0     │ │
│  │  = 0.75     │  │  = 0.75     │  │                         │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
│         │                │                      │              │
│         │     模型更新    │                      │              │
│         │                │                      │              │
│         └────────────────┼──────────────────────┘              │
│                          ▼                                     │
│                 ┌────────────────┐                             │
│                 │   FreqFed防御   │                             │
│                 │  ┌──────────┐  │                             │
│                 │  │ HDBSCAN  │  │                             │
│                 │  │ 聚类检测 │  │                             │
│                 │  └──────────┘  │                             │
│                 └───────┬────────┘                             │
│                         │                                      │
│                         ▼                                      │
│                 ┌────────────────┐                             │
│                 │   FedAvg聚合    │                             │
│                 │  加权平均更新   │                             │
│                 └───────┬────────┘                             │
│                         │                                      │
│                         ▼                                      │
│                 ┌────────────────┐                             │
│                 │    全局模型     │                             │
│                 └────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 攻击目标

| 目标 | 说明 | 评估指标 |
|------|------|----------|
| **后门植入** | 模型对带触发器的输入预测为目标类 | ASR ≥ 85% |
| **防御绕过** | 恶意更新不被 FreqFed 检测 | Bypass ≥ 70% |
| **主任务可用** | 模型对正常输入仍有准确预测 | ACC ≥ 60% |

---

## 五、实验配置

### 5.1 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_clients` | 10 | 总客户端数 |
| `poison_ratio` | 0.2 | 恶意客户端比例 (2/10) |
| `poison_rate` | 0.75 | 恶意客户端毒化比例 |
| `scaling_factor` | 5.5 | 模型替换放大因子 |
| `epsilon` | 0.1 | 触发器注入强度 |
| `target_label` | 0 | 后门目标类 |
| `num_rounds` | 30 | 联邦学习轮数 |
| `local_epochs` | 3 | 本地训练轮数 |
| `learning_rate` | 0.01 | 学习率 |
| `backdoor_boost_weight` | 0.3 | 后门增强损失权重 |

### 5.2 数据集

| 数据集 | 类别数 | 图像尺寸 | 训练集大小 | 测试集大小 |
|--------|--------|----------|------------|------------|
| CIFAR-10 | 10 | 32×32×3 | 50,000 | 10,000 |
| CIFAR-100 | 100 | 32×32×3 | 50,000 | 10,000 |

### 5.3 模型

- **ResNet-18**：主要实验模型
- 输入：32×32 RGB 图像
- 输出：10/100 类别概率

---

## 六、核心创新点总结

ANB 针对 FreqFed 防御的四个关键优化：

| 创新点 | 解决的问题 | 技术手段 |
|--------|------------|----------|
| **频率分散** | 所有恶意客户端特征相似，易被聚类 | 不同客户端使用不同频率位置 |
| **动态相位** | 特征在训练过程中一致，易被检测 | 每轮训练相位变化 |
| **星云扩散** | 点状触发器频域特征明显 | 高斯扩散降低可检测性 |
| **双域注入** | 纯频域注入在某些区域效果差 | 根据图像内容自适应选择注入方式 |

这些设计使恶意更新在频域特征空间中分散、动态、低可检测性，从而有效绕过 FreqFed 的聚类防御。

---

## 七、参考文献

1. **FreqFed**: Frequency-Based Defense for Federated Learning. NDSS 2024.
2. **Bagdasaryan et al.**: How to Backdoor Federated Learning. AISTATS 2020. (Model Replacement Attack)
3. **FIBA**: Frequency-Based Backdoor Attack. (FIXED 策略的基础)
