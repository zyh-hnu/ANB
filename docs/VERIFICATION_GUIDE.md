# ANB (Adaptive Nebula Backdoor) 验证指南

## 验证流程

### 步骤 1: 原子验证（Atomic Verification）

**目的**: 验证ANB的核心机制是否正常工作

**运行命令**:
```bash
python analysis/anb_atomic_verification.py
```

**预期结果**:
1. **Phase Scheduling**: 在不同轮次应显示不同的sigma和phase值
   - Round 5: sigma=0.8, 固定phase
   - Round 25: sigma=0.8, 随机phase（4选1）
   - Round 40: sigma=1.5, 随机phase（8选1）

2. **Dual-Domain Routing**:
   - 纹理图像: Frequency Routing Ratio > 0.5
   - 平坦图像: Spatial Routing Ratio > 0.5

3. **Frequency Sharding**:
   - 应显示10个客户端使用不同的频率中心
   - Diversity Ratio应接近100%

4. **Imperceptibility**:
   - 所有轮次的PSNR应 > 30 dB

**输出文件**:
- `./results/anb_verification/trigger_evolution.png` - 触发器演化可视化
- `./results/anb_verification/summary_report.png` - 综合报告

---

### 步骤 2: 基础功能测试（Self-Test）

**运行命令**:
```bash
python core/attacks.py
```

**预期输出**:
```
Initializing ANB Ultimate Self-Test...

[Stage 1: Stabilization (Round 5)]
-> Trigger Generated. Nebula Sigma ~ 0.8

[Stage 2: Expansion (Round 25)]

[Stage 3: Max Chaos (Round 40)]

Final Stats @ Round 40:
  Texture Region Perturbation (Freq): XX.XX (Target ~ Epsilon * 1.5 * 255)
  Flat Region Perturbation (Spatial): XX.XX (Target Low, visible only in corner)

Phase Strategy Check (should vary): [...]

✓ ANB Self-Test Completed Successfully!
```

**验证要点**:
- 纹理区域扰动 ≈ 0.1 × 1.5 × 255 = 38.25
- 平坦区域扰动较低（仅角落区域）
- Phase列表应在Round 40显示变化

---

### 步骤 3: 完整联邦学习实验

**3.1 无防御实验（验证ASR）**

编辑 `config.py`:
```python
DEFENSE_ENABLED = False
FREQ_STRATEGY = 'DISPERSED'
NUM_ROUNDS = 50
EPSILON = 0.1
```

运行:
```bash
python main.py
```

**成功标准**:
- Final ASR (Single Trigger) > 90%
- Final ASR (Multi-Trigger) > 90%
- Final Clean Accuracy > 70%

**3.2 有防御实验（验证绕过）**

编辑 `config.py`:
```python
DEFENSE_ENABLED = True
DEFENSE_METHOD = 'hdbscan'
```

运行:
```bash
python main.py
```

**成功标准**:
- 恶意客户端应分散到良性簇中（不被聚类分离）
- ASR仍保持 > 80%

---

### 步骤 4: 对比实验（ANB vs. 原始方法）

**方法1: 使用备份的原始attacks.py**

```bash
# 恢复原始方法
cp core/attacks.py.backup core/attacks_anb.py
cp core/attacks.py.backup core/attacks.py

# 运行原始方法
python main.py

# 切换回ANB
cp core/attacks_anb.py core/attacks.py

# 运行ANB方法
python main.py
```

**对比指标**:
1. **ASR**: ANB应与原始方法相当或更高
2. **Defense Evasion**: ANB应显著优于原始方法（查看聚类结果）
3. **Imperceptibility**: 两者应相似（PSNR > 30dB）

---

## 快速验证检查清单

### ✅ 代码适配检查

- [x] `core/attacks.py` - ANB实现已替换
- [x] `federated/client.py` - 添加了`current_round`参数传递
- [x] `federated/server.py` - 调用`client.train()`时传递`current_round`
- [x] `data/dataset.py` - BackdoorDataset使用FrequencyBackdoor（兼容层）

### ✅ 核心机制验证

运行以下Python代码快速测试:

```python
from core.attacks import AdaptiveNebulaBackdoor
import numpy as np

# 测试1: Phase Scheduling
backdoor = AdaptiveNebulaBackdoor(client_id=0)
backdoor.set_round(5)
print(f"Round 5 sigma: {backdoor._get_adaptive_sigma()}")  # 应输出 0.8

backdoor.set_round(40)
print(f"Round 40 sigma: {backdoor._get_adaptive_sigma()}")  # 应输出 1.5

# 测试2: Trigger Generation
dummy_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
poisoned, label = backdoor(dummy_img, 5)
diff = np.mean(np.abs(poisoned.astype(float) - dummy_img.astype(float)))
print(f"Mean perturbation: {diff:.2f}")  # 应在 5-20 范围内

print("✓ All quick tests passed!")
```

### ✅ 完整实验验证

**最小验证实验** (快速测试, 10 rounds):

编辑 `config.py`:
```python
NUM_ROUNDS = 10
NUM_CLIENTS = 5
LOCAL_EPOCHS = 2
```

运行: `python main.py`

预期时间: 5-10分钟（GPU）

**标准验证实验** (完整验证, 50 rounds):

恢复原始配置运行完整实验。

---

## 关键输出文件

实验完成后检查以下文件:

1. **模型权重**: `./results/model_OURS_DISPERSED_defense_*.pth`
2. **训练历史**: `./results/history_OURS_DISPERSED_defense_*.json`
3. **客户端权重**: `./results/weights/client_weights_round_*.pkl`
4. **验证报告**: `./results/anb_verification/*.png`

---

## 常见问题排查

### Q1: 导入错误 `ModuleNotFoundError`

**解决**: 确保已安装所有依赖
```bash
pip install torch torchvision opencv-python numpy scikit-learn hdbscan matplotlib
```

### Q2: CUDA内存不足

**解决**: 编辑 `config.py`
```python
BATCH_SIZE = 16  # 减小batch size
NUM_CLIENTS = 5   # 减少客户端数量
```

### Q3: ASR过低（< 50%）

**可能原因**:
- Epsilon太小: 增加到0.15
- Rounds太少: 增加到100
- 防御太强: 先关闭防御测试

### Q4: 聚类未能分离恶意客户端（防御失效）

**这是预期行为!** ANB的目标就是绕过聚类防御。如果恶意客户端被成功分离，说明需要调整:
- 增加频率分片多样性
- 调整sigma参数

---

## 成功验证的标志

✅ **原子验证通过**: 所有6个验证项通过
✅ **基础功能正常**: Self-test无报错
✅ **无防御ASR > 90%**: 攻击有效性验证
✅ **有防御ASR > 80%**: 防御绕过验证
✅ **可视化生成**: 所有图像文件正常生成

完成以上验证后，可以撰写论文实验章节！
