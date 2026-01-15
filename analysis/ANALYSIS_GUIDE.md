# SAFB 分析工具使用指南

本指南详细说明了 SAFB 项目中所有分析工具的功能、使用方法和分析要点。

## 目录

1. [ANB 原子验证](#1-anb-原子验证-anb_atomic_verificationpy)
2. [频域特性验证](#2-频域特性验证-verify_frequency_propertiespy)
3. [防御聚类测试](#3-防御聚类测试-test_defense_clusteringpy)
4. [聚类可视化](#4-聚类可视化-visualize_clusterspy)
5. [不可感知性评估](#5-不可感知性评估-evaluate_imperceptibilitypy)
6. [频域残差分析](#6-频域残差分析-frequency_residual_analysispy)
7. [GradCAM 检查](#7-gradcam-检查-gradcam_checkpy)
8. [综合可视化生成](#8-综合可视化生成-create_visualizationspy)

---

## 1. ANB 原子验证 (`anb_atomic_verification.py`)

### 功能
验证 ANB (Adaptive Nebula Backdoor) 攻击的四大核心机制：

1. **相位动态混沌控制器** (Phased Dynamic Chaos)
   - 验证三个训练阶段的相位调度行为
   - Stage 1 (Round 5): 稳定阶段 - 确定性相位
   - Stage 2 (Round 25): 扩展阶段 - 主相位池随机选择
   - Stage 3 (Round 40): 最大混沌阶段 - 全相位池随机选择

2. **自适应 Sigma 过渡** (Adaptive Sigma)
   - 验证 Sigma 从 0.8 (锐利) 到 1.5 (模糊) 的过渡
   - 检查能量补偿因子的正确计算

3. **双域路由行为** (Dual-Domain Routing)
   - 验证基于图像复杂度的路由决策
   - 高复杂度区域 → 频域注入
   - 低复杂度区域 → 空域注入

4. **多客户端频率分片** (Frequency Sharding)
   - 验证不同客户端使用不同的频率模式
   - 确保频率多样性以绕过聚类检测

### 是否需要训练模型
❌ **不需要**

### 使用方法
```bash
# 直接运行
python analysis/anb_atomic_verification.py
```

### 输出结果
- **控制台输出**：
  - 每个阶段的相位调度验证结果
  - Sigma 和补偿因子数值
  - 双域路由比例统计
  - 频率分片分配表
  - PSNR 和 MSE 统计
- **可视化文件**：
  - `./results/anb_verification/trigger_evolution.png` - 触发器演化图
  
  - `./results/anb_verification/summary_report.png` - 综合摘要报告

### 分析要点

#### 1. 相位调度验证
- **Stage 1 (Round 5)**: 所有样本的相位应该相同（确定性）
- **Stage 2 (Round 25)**: 相位应在 4 个主相位中随机选择
- **Stage 3 (Round 40)**: 相位应在 8 个相位中随机选择

#### 2. 自适应 Sigma 验证
- Round 5: Sigma ≈ 0.8（锐利，类似星点）
- Round 40: Sigma ≈ 1.5（模糊，类似星云）
- 补偿因子应随 Sigma 增大而增大

#### 3. 双域路由验证
- **纹理图像**: 频域路由比例 > 空域路由比例
- **平坦图像**: 空域路由比例 > 频域路由比例
- 路由决策应基于图像的局部复杂度

#### 4. 频率分片验证
- 不同客户端应使用不同的频率中心
- 频率多样性比率应 > 80%
- 频率应分布在 [2, 7] 范围内

---

## 2. 频域特性验证 (`verify_frequency_properties.py`)

### 功能
验证触发器的频域纯度，分析 FIXED 和 ANB 两种策略的频域特性差异。

**注意**: 此文件需要 `core.attacks` 中的辅助函数：
- `soft_edge_extraction()` - 边缘提取
- `get_frequency_pattern()` - 频率模式生成
- `inject_frequency_trigger()` - 频率触发器注入

### 是否需要训练模型
❌ **不需要**

### 使用方法
```bash
# 直接运行
python analysis/verify_frequency_properties.py
```

### 输出结果
- **控制台输出**：
  - 每个客户端的频率纯度（Frequency Purity）
  - PSNR 值
  - 平均统计数据
- **可视化文件**：
  - `./results/frequency_analysis_client0.png` - 频谱分析图
  - `./results/strategy_comparison.png` - FIXED vs ANB 对比

### 分析要点

#### 1. Frequency Purity
- 目标值：> 50%
- FIXED 策略：所有客户端纯度应相似
- ANB 策略：不同客户端纯度应有差异

#### 2. PSNR
- 目标值：> 30 dB
- 两种策略的 PSNR 应相近（隐蔽性相当）

#### 3. 频谱图分析
- FIXED：能量集中在特定频率点
- ANB：能量分散在多个频率点

---

## 3. 防御聚类测试 (`test_defense_clustering.py`)

### 功能
测试 FreqFed 防御的检测能力，评估 FIXED 和 ANB 策略的防御绕过效果。

### 是否需要训练模型
⚠️ **可选**
- 使用 `--use-synthetic-weights`：不需要训练模型（使用合成权重）
- 使用真实权重：需要先运行 `main.py` 生成客户端权重

### 使用方法

**方法 1：使用合成权重（快速验证）**
```bash
python analysis/test_defense_clustering.py
```

**方法 2：使用真实权重（最终评估）**
```bash
# 步骤 1：运行联邦学习训练
python main.py

# 步骤 2：使用生成的权重进行聚类分析
python analysis/test_defense_clustering.py --use-real-weights --weights-dir ./results/weights
```

### 输出结果
- **控制台输出**：
  - FIXED 策略的聚类结果
  - ANB 策略的聚类结果
  - 防御召回率（Recall）
  - 防御有效性评估
- **可视化文件**：
  - `./results/defense_visualization/clustering_fixed.png` - FIXED 聚类图
  - `./results/defense_visualization/clustering_anb.png` - ANB 聚类图
  - `./results/defense_visualization/tsne_fixed.png` - FIXED t-SNE 图
  - `./results/defense_visualization/tsne_anb.png` - ANB t-SNE 图

### 分析要点

#### 1. FIXED 策略
- 恶意客户端应被聚类到同一簇
- 防御召回率应 > 60%（防御有效）
- t-SNE 可视化显示明显的簇分离

#### 2. ANB 策略
- 恶意客户端应分散到多个簇
- 防御召回率应 < 50%（防御被绕过）
- t-SNE 可视化显示恶意客户端混入良性簇

#### 3. 聚类可视化
- 蓝色点：良性客户端
- 红色三角形：恶意客户端
- 观察恶意客户端是否混入良性簇

---

## 4. 聚类可视化 (`visualize_clusters.py`)

### 功能
可视化客户端在 DCT 特征空间的分布，直观展示防御绕过效果。

### 是否需要训练模型
⚠️ **可选**
- 使用 `--use-real-weights=False`：不需要训练模型（使用合成权重）
- 使用真实权重：需要先运行 `main.py` 生成客户端权重

### 使用方法

**方法 1：使用合成权重**
```bash
python analysis/visualize_clusters.py
```

**方法 2：使用真实权重**
```bash
# 步骤 1：运行联邦学习训练
python main.py

# 步骤 2：可视化真实权重
python analysis/visualize_clusters.py --use-real-weights --weights-dir ./results/weights
```

### 输出结果
- **控制台输出**：
  - 聚类统计信息
  - 每个簇中的客户端数量
  - 恶意客户端分布
- **可视化文件**：
  - `./results/defense_visualization/clustering_*.png` - PCA 降维后的 2D 散点图

### 分析要点

#### 1. PCA 降维
- X 轴：PCA Component 1（解释方差百分比）
- Y 轴：PCA Component 2（解释方差百分比）

#### 2. 簇分布
- 良性客户端（蓝色圆点）应形成大簇
- 恶意客户端（红色三角形）：
  - FIXED：形成独立小簇（防御成功）
  - ANB：混入良性簇（防御绕过）

#### 3. 防御效果评估
- 所有恶意客户端在同一簇且无良性客户端 → 防御成功
- 恶意客户端分散在多个簇 → 防御被绕过

---

## 5. 不可感知性评估 (`evaluate_imperceptibility.py`)

### 功能
量化评估触发器的隐蔽性，计算 PSNR、SSIM 和 LPIPS 指标。

对比两种策略：
- **FIBA (FIXED)**: 静态频率触发器（基线）
- **ANB (OURS)**: 动态星云触发器（我们的方法）

### 是否需要训练模型
❌ **不需要**

### 使用方法
```bash
# 直接运行
python analysis/evaluate_imperceptibility.py
```

### 输出结果

- **控制台输出**：
  - PSNR（峰值信噪比）
  - SSIM（结构相似性）
  - LPIPS（感知相似性）
  - 两种策略的对比结果
- **文本文件**：
  - `./results/imperceptibility_results.txt` - 详细数值结果

### 分析要点

#### 1. PSNR (Peak Signal-to-Noise Ratio)
- 目标值：> 30 dB
- 值越高，触发器越不可感知
- ANB 可能略低于 FIBA（因为星云扩散覆盖更多像素）

#### 2. SSIM (Structural Similarity Index)
- 目标值：> 0.95
- 值越接近 1，结构相似性越好
- 两种策略应相近

#### 3. LPIPS (Learned Perceptual Image Patch Similarity)
- 目标值：< 0.1
- 值越低，人眼感知差异越小
- ANB 应优化感知隐蔽性

#### 4. 结果解读
- PSNR: ANB 可能略低，但应在可接受范围
- SSIM: 两种策略应相近
- LPIPS: ANB 应不显著高于 FIBA

---

## 6. 频域残差分析 (`frequency_residual_analysis.py`)

### 功能
分析干净图像和被攻击图像之间的频谱残差，比较 FIXED 和 ANB 策略的频域特性。

### 是否需要训练模型
❌ **不需要**

### 使用方法
```bash
# 直接运行
python analysis/frequency_residual_analysis.py
```

### 输出结果

- **控制台输出**：
  - 各频段的能量分布（低频、中频、高频）
  - FIXED vs ANB 的能量对比
- **可视化文件**：
  - `./results/frequency_analysis/residual_fixed.png` - FIXED 频域残差
  - `./results/frequency_analysis/residual_anb.png` - ANB 频域残差
  - `./results/frequency_analysis/energy_comparison.png` - 能量对比图
  - `./results/frequency_analysis/residual_comparison.png` - 残差对比图
  - `./results/frequency_analysis/frequency_analysis_summary.txt` - 数值结果

### 分析要点

#### 1. 频段能量分布
- FIXED：能量集中在特定频段
- ANB：能量分散在多个频段

#### 2. 能量对比
- ANB 的低频能量应低于 FIXED
- ANB 的中频能量应高于 FIXED

#### 3. 残差可视化
- FIXED：明显的热点（能量集中）
- ANB：分散的星云状分布

---

## 7. GradCAM 检查 (`gradcam_check.py`)

### 功能
验证触发器的语义感知特性，检查触发器是否与对象轮廓集成。

### 是否需要训练模型
✅ **需要加载训练后的模型参数**

### 使用方法

**步骤 1：训练模型**
```bash
# 修改 config.py 设置攻击模式
# ATTACK_MODE = 'OURS'  # 或 'FIBA'

# 运行联邦学习训练
python main.py
```

**步骤 2：修改 gradcam_check.py 加载模型**
```python
# 在 main() 函数中，找到这行代码：
# model.load_state_dict(torch.load('global_model.pth'))

# 修改为实际的模型路径：
model.load_state_dict(torch.load('./results/model_OURS_ANB_defense_True.pth'))
```

**步骤 3：运行 GradCAM 检查**
```bash
python analysis/gradcam_check.py
```

### 输出结果
- **控制台输出**：
  - 注意力保留率（Focus Retention）
  - 验证结果（Success/Failure）
- **可视化文件**：
  - `gradcam_comparison.png` - GradCAM 热力图对比

### 分析要点

#### 1. 注意力保留率
- 目标值：> 70%
- 触发器不应显著改变模型的注意力区域

#### 2. 热力图对比
- 原始图像：注意力集中在对象上
- 被攻击图像：注意力仍应集中在对象上
- 触发器应集成在边缘区域，不干扰主要注意力

#### 3. 验证结果
- Success：触发器与对象轮廓集成
- Failure：触发器干扰对象注意力

---

## 8. 综合可视化生成 (`create_visualizations.py`)

### 功能
生成学术论文级别的图表，包括触发器生成流程、多客户端触发器、频率策略对比、防御绕过概念图。

### 是否需要训练模型
❌ **不需要**

### 使用方法
```bash
# 直接运行
python analysis/create_visualizations.py
```

### 输出结果
- **可视化文件**：
  - `./results/trigger_pipeline.png` - 触发器生成流程图
  - `./results/multi_client_triggers.png` - 多客户端触发器展示
  - `./results/frequency_comparison.png` - FIXED vs ANB 对比
  - `./results/defense_evasion_concept.png` - 防御绕过概念图

### 分析要点

#### 1. 触发器生成流程图

- 展示：原始图像 → 边缘提取 → 频率模式 → 最终触发器
- 用于论文中解释攻击方法

#### 2. 多客户端触发器
- 展示不同客户端的频率多样性
- 每个客户端使用不同的频率中心

#### 3. 频率策略对比
- FIXED：所有客户端使用相同频率
- ANB：不同客户端使用不同频率

#### 4. 防御绕过概念图
- FIXED：恶意客户端形成独立簇（防御有效）
- ANB：恶意客户端混入良性簇（防御绕过）

---

## 模型参数加载说明

### 需要加载模型参数的工具

以下工具需要加载训练后的模型参数：

1. **`gradcam_check.py`** - 需要 ResNet-18 模型参数
2. **`test_defense_clustering.py`** - 需要客户端权重（可选）
3. **`visualize_clusters.py`** - 需要客户端权重（可选）

### 模型参数文件位置

训练完成后，模型参数保存在：
```
./results/
├── model_OURS_ANB_defense_True.pth      # ANB 全局模型参数
├── model_FIBA_FIXED_defense_True.pth          # FIBA 模型参数
└── weights/
    ├── client_weights_round_25.pkl            # 第 25 轮的客户端权重
    └── client_weights_round_50.pkl            # 第 50 轮的客户端权重
```

### 如何加载模型参数

#### 方法 1：加载全局模型（用于 GradCAM）

```python
import torch
from models.resnet import ResNet18

# 1. 初始化模型
model = ResNet18(num_classes=10)

# 2. 加载训练后的参数
model_path = './results/model_OURS_ANB_defense_True.pth'
model.load_state_dict(torch.load(model_path))

# 3. 设置为评估模式
model.eval()

# 4. 使用模型进行推理
with torch.no_grad():
    output = model(input_tensor)
```

#### 方法 2：加载客户端权重（用于聚类分析）

```python
import pickle
import os

# 1. 指定权重文件路径
weights_dir = './results/weights'
round_num = 50
weights_file = os.path.join(weights_dir, f'client_weights_round_{round_num}.pkl')

# 2. 加载客户端权重
with open(weights_file, 'rb') as f:
    data = pickle.load(f)

# 3. 提取数据
client_weights = data['client_weights']          # 客户端权重列表
malicious_indices = data['malicious_indices']      # 恶意客户端索引
metadata = data                                   # 元数据（轮次、客户端数等）

# 4. 使用权重进行聚类分析
from core.defenses import cluster_client_models

labels, features = cluster_client_models(
    client_weights,
    method='hdbscan'
)
```

### 检查模型参数加载是否正确

#### 检查全局模型加载

```python
import torch
from models.resnet import ResNet18

# 加载模型
model = ResNet18(num_classes=10)
model_path = './results/model_OURS_ANB_defense_True.pth'

try:
    model.load_state_dict(torch.load(model_path))
    print("✓ 模型参数加载成功")
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
except FileNotFoundError:
    print("✗ 模型文件不存在，请先运行 python main.py")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
```

#### 检查客户端权重加载

```python
import pickle
import os

weights_file = './results/weights/client_weights_round_50.pkl'

try:
    with open(weights_file, 'rb') as f:
        data = pickle.load(f)

    print("✓ 客户端权重加载成功")
    print(f"  轮次: {data['round']}")
    print(f"  客户端数量: {data['num_clients']}")
    print(f"  恶意客户端: {data['malicious_indices']}")
    print(f"  权重数量: {len(data['client_weights'])}")

    # 检查权重格式
    first_client = data['client_weights'][0]
    print(f"  第一个客户端的层数: {len(first_client)}")
    print(f"  第一层名称: {list(first_client.keys())[0]}")

except FileNotFoundError:
    print("✗ 权重文件不存在，请先运行 python main.py")
except Exception as e:
    print(f"✗ 权重加载失败: {e}")
```

---

## 分析流程建议

### 阶段 1：快速验证（无需训练）

**目标**：快速验证攻击方法的基本特性

```bash
# 1. ANB 原子验证
python analysis/anb_atomic_verification.py

# 2. 频域特性验证
python analysis/verify_frequency_properties.py

# 3. 不可感知性评估
python analysis/evaluate_imperceptibility.py

# 4. 频域残差分析
python analysis/frequency_residual_analysis.py

# 5. 生成可视化图表
python analysis/create_visualizations.py
```

**预期结果**：
- ANB 四大机制验证通过
- 频域纯度 > 50%
- PSNR > 30 dB
- SSIM > 0.95
- LPIPS < 0.1

---

### 阶段 2：防御绕过验证（可选训练）

**目标**：验证防御绕过能力

```bash
# 1. 运行联邦学习训练（生成模型参数）
python main.py

# 2. 防御聚类测试（使用真实权重）
python analysis/test_defense_clustering.py --use-real-weights --weights-dir ./results/weights

# 3. 聚类可视化
python analysis/visualize_clusters.py --use-real-weights --weights-dir ./results/weights
```

**预期结果**：
- ANB 策略的防御召回率 < 50%
- 恶意客户端混入良性簇
- 防御被成功绕过

---

### 阶段 3：深度分析（需要训练）

**目标**：深入分析触发器的语义感知特性

```bash
# 1. 修改 gradcam_check.py 加载模型参数
#    找到这行：model.load_state_dict(torch.load('global_model.pth'))
#    修改为：model.load_state_dict(torch.load('./results/model_OURS_ANB_defense_True.pth'))

# 2. 运行 GradCAM 检查
python analysis/gradcam_check.py
```

**预期结果**：
- 注意力保留率 > 70%
- 触发器与对象轮廓集成
- 不干扰主要注意力区域

---

### 完整分析流程图

```
开始
  ↓
[阶段 1：快速验证]
  ├─ anb_atomic_verification.py
  ├─ verify_frequency_properties.py
  ├─ evaluate_imperceptibility.py
  ├─ frequency_residual_analysis.py
  └─ create_visualizations.py
  ↓
是否需要验证防御绕过？
  ├─ 是 → [阶段 2：防御绕过验证]
  │       ├─ python main.py
  │       ├─ test_defense_clustering.py
  │       └─ visualize_clusters.py
  │       ↓
  │   是否需要深度分析？
  │   ├─ 是 → [阶段 3：深度分析]
  │   │       └─ gradcam_check.py
  │   │       ↓
  └─ 否 → 结束
结束
```

---

## 常见问题

### Q1: 运行分析工具时提示找不到模型文件怎么办？

**A**: 请先运行 `python main.py` 进行训练，生成模型参数文件。

```bash
# 检查模型文件是否存在
ls ./results/model_*.pth

# 如果不存在，运行训练
python main.py
```

### Q2: 如何选择使用合成权重还是真实权重？

**A**:
- **合成权重**：适合快速验证，不需要训练，运行速度快
- **真实权重**：适合最终评估，反映实际训练效果，需要先训练

### Q3: GradCAM 检查失败怎么办？

**A**: 检查以下几点：
1. 模型文件路径是否正确
2. 模型是否已加载参数
3. 目标层是否正确（`model.layer4[-1].conv2`）

### Q4: 频域纯度低于 50% 怎么办？

**A**: 可能的原因：
1. 频率池选择不当（存在谐波重叠）
2. 边缘掩码抑制过强
3. 注入强度 epsilon 过大

### Q5: ANB 原子验证失败怎么办？

**A**: 检查以下几点：
1. `AdaptiveNebulaBackdoor` 类是否正确导入
2. 训练轮次设置是否正确
3. 频率分片配置是否完整

### Q6: 如何理解 ANB 的四大机制？

**A**:
1. **相位动态混沌**: 根据训练阶段动态调整相位策略
2. **自适应 Sigma**: 从锐利到模糊的过渡，平衡可学习性和隐蔽性
3. **双域路由**: 根据图像复杂度智能选择频域或空域注入
4. **频率分片**: 不同客户端使用不同频率，绕过聚类检测

---

## 攻击方法对比

| 特性 | FIBA (FIXED) | ANB (OURS) |
|------|----------------|-------------|
| 频率策略 | 静态固定频率 | 动态分片频率 |
| 相位控制 | 固定相位 | 动态相位调度 |
| Sigma | 固定值 | 自适应过渡 |
| 注入域 | 仅频域 | 双域路由 |
| 防御绕过 | 易被检测 | 难以检测 |
| 可学习性 | 高 | 高（通过补偿） |
| 隐蔽性 | 高 | 高（通过路由） |

---

## 总结

本指南提供了 SAFB 项目中所有分析工具的详细说明。建议按照以下顺序使用：

1. **快速验证阶段**：验证基本机制和特性
2. **防御绕过验证**：验证实际防御绕过能力
3. **深度分析**：验证语义感知特性

每个工具都有明确的使用方法和分析要点，请根据研究需求选择合适的工具。

如有问题，请参考常见问题部分或查看具体工具的源代码注释。
