# Google Colab 运行指南

本指南帮助你将 ANB 项目部署到 Google Colab 上运行，适合没有 GPU 资源的用户。

---

## 一、快速开始

### 1.1 打开 Colab 笔记本

1. 登录 [Google Colab](https://colab.research.google.com/)
2. 点击「文件」→「新建笔记本」

### 1.2 设置 GPU 运行时

1. 点击「运行时」→「更改运行时类型」
2. 硬件加速器选择 **T4 GPU**
3. 点击「保存」

### 1.3 验证 GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 二、项目安装

### 2.1 克隆项目

```python
# 克隆仓库
!git clone https://github.com/zyh-hnu/ANB.git
%cd ANB
```

### 2.2 安装依赖

```python
# 安装基础依赖
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
!pip install -q opencv-python-headless scikit-image hdbscan lpips tqdm pyyaml seaborn
```

### 2.3 验证安装

```python
# 测试导入
import torch
import torchvision
import hdbscan
import lpips

print("✓ 所有依赖安装成功")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## 三、运行实验

### 3.1 快速冒烟测试（推荐首次运行）

```python
# 运行基础实验（约 5-10 分钟）
!python main.py \
    --num-rounds 10 \
    --local-epochs 3 \
    --poison-rate 0.75 \
    --scaling-factor 5.5 \
    --backdoor-boost-weight 0.3 \
    --defense-enabled 1 \
    --defense-method hdbscan
```

### 3.2 完整实验（30 轮）

```python
# 完整实验配置
!python main.py \
    --num-rounds 30 \
    --local-epochs 3 \
    --poison-rate 0.75 \
    --scaling-factor 5.5 \
    --backdoor-boost-weight 0.3 \
    --seed 42
```

### 3.3 消融实验

```python
# FIXED 基线（对比实验）
!python main.py \
    --freq-strategy FIXED \
    --num-rounds 30 \
    --poison-rate 0.75 \
    --scaling-factor 5.5

# 无防御场景（攻击上界）
!python main.py \
    --defense-enabled 0 \
    --num-rounds 20 \
    --poison-rate 1.0 \
    --scaling-factor 5.0
```

### 3.4 自定义参数实验

```python
# 方案一：参数扫描
configs = [
    {"poison_rate": 0.70, "scaling_factor": 6.0, "boost": 0.3},
    {"poison_rate": 0.75, "scaling_factor": 5.5, "boost": 0.3},
    {"poison_rate": 0.80, "scaling_factor": 5.0, "boost": 0.3},
    {"poison_rate": 0.75, "scaling_factor": 5.0, "boost": 0.5},
]

for i, cfg in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Running config {i+1}/{len(configs)}")
    print(f"{'='*60}")
    
    !python main.py \
        --num-rounds 30 \
        --local-epochs 3 \
        --poison-rate {cfg['poison_rate']} \
        --scaling-factor {cfg['scaling_factor']} \
        --backdoor-boost-weight {cfg['boost']}
```

---

## 四、参数说明

### 4.1 攻击参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--poison-rate` | 1.0 | 恶意客户端毒化比例 (0~1) |
| `--scaling-factor` | 5.0 | 模型替换放大因子 |
| `--backdoor-boost-weight` | 0.3 | 后门增强损失权重 |
| `--epsilon` | 0.1 | 触发器注入强度 |
| `--target-label` | 0 | 后门目标类 |
| `--freq-strategy` | ANB | 攻击策略 (ANB/FIXED) |

### 4.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-rounds` | 50 | 联邦学习轮数 |
| `--local-epochs` | 5 | 本地训练轮数 |
| `--learning-rate` | 0.01 | 学习率 |
| `--batch-size` | 32 | 批次大小 |
| `--num-clients` | 10 | 客户端数量 |
| `--seed` | 42 | 随机种子 |

### 4.3 防御参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--defense-enabled` | 1 | 是否启用防御 (0/1) |
| `--defense-method` | hdbscan | 防御方法 (hdbscan/kmeans/dbscan) |

---

## 五、结果查看

### 5.1 查看结果文件

```python
import os
import json

# 查看结果目录
!ls -la results/

# 查看训练历史
if os.path.exists('results/history_anb_freqfed.json'):
    with open('results/history_anb_freqfed.json', 'r') as f:
        history = json.load(f)
    
    print("\n=== Final Results ===")
    print(f"Final ACC: {history['test_acc'][-1]:.2%}")
    print(f"Final ASR: {history['test_asr'][-1]:.2%}")
    if history.get('test_asr_multi'):
        print(f"Final ASR (Multi): {history['test_asr_multi'][-1]:.2%}")
    if history.get('defense_bypass_rate'):
        print(f"Final Bypass Rate: {history['defense_bypass_rate'][-1]:.2%}")
```

### 5.2 可视化结果

```python
import matplotlib.pyplot as plt
import json

# 加载结果
with open('results/history_anb_freqfed.json', 'r') as f:
    history = json.load(f)

# 创建图表
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ACC 曲线
axes[0].plot(history['test_acc'], 'b-', linewidth=2)
axes[0].set_xlabel('Round')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Clean Accuracy')
axes[0].grid(True, alpha=0.3)

# ASR 曲线
axes[1].plot(history['test_asr'], 'r-', linewidth=2, label='Single')
if history.get('test_asr_multi'):
    axes[1].plot(history['test_asr_multi'], 'r--', linewidth=2, label='Multi')
axes[1].set_xlabel('Round')
axes[1].set_ylabel('ASR (%)')
axes[1].set_title('Attack Success Rate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Bypass Rate 曲线
if history.get('defense_bypass_rate'):
    axes[2].plot(history['defense_bypass_rate'], 'g-', linewidth=2)
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Bypass Rate (%)')
    axes[2].set_title('Defense Bypass Rate')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/experiment_results.png', dpi=150)
plt.show()
```

### 5.3 下载结果

```python
# 打包结果
!zip -r results.zip results/

# 下载到本地
from google.colab import files
files.download('results.zip')
```

---

## 六、常见问题

### 6.1 CUDA 内存不足

```python
# 减小批次大小
!python main.py --batch-size 16 --num-rounds 30

# 或减少本地训练轮数
!python main.py --local-epochs 2 --num-rounds 30
```

### 6.2 运行时间过长

```python
# 减少训练轮数进行快速验证
!python main.py --num-rounds 10 --local-epochs 2
```

### 6.3 Colab 断开连接

Colab 免费版有时间限制，建议：
1. 定期保存结果到 Google Drive
2. 使用检查点恢复训练

```python
# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存结果到 Drive
!cp -r results/ /content/drive/MyDrive/ANB_results/
```

### 6.4 依赖冲突

```python
# 重启运行时后重新安装
import os
os.kill(os.getpid(), 9)
```

---

## 七、完整 Notebook 模板

将以下内容复制到 Colab 笔记本中：

```python
# === Cell 1: 环境设置 ===
!nvidia-smi
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# === Cell 2: 克隆项目 ===
!git clone https://github.com/zyh-hnu/ANB.git
%cd ANB

# === Cell 3: 安装依赖 ===
!pip install -q opencv-python-headless scikit-image hdbscan lpips tqdm pyyaml seaborn

# === Cell 4: 运行实验 ===
!python main.py \
    --num-rounds 30 \
    --local-epochs 3 \
    --poison-rate 0.75 \
    --scaling-factor 5.5 \
    --backdoor-boost-weight 0.3 \
    --seed 42

# === Cell 5: 查看结果 ===
import json
with open('results/history_anb_freqfed.json', 'r') as f:
    h = json.load(f)
print(f"ACC: {h['test_acc'][-1]:.2%}")
print(f"ASR: {h['test_asr'][-1]:.2%}")
print(f"Bypass: {h['defense_bypass_rate'][-1]:.2%}")

# === Cell 6: 下载结果 ===
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

---

## 八、预期结果

成功运行后，应达到以下指标：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **ACC** | ≥ 60% | 主任务准确率 |
| **ASR** | ≥ 85% | 攻击成功率 |
| **Bypass** | ≥ 70% | 防御绕过率 |

如果结果不理想，可调整参数：
- ACC 过低 → 降低 `poison_rate`
- ASR 过低 → 提高 `scaling_factor` 或 `backdoor_boost_weight`
- Bypass 过低 → 已达到 100% 无需调整
