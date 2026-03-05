# SAFB 项目结构与可插拔性说明

本文档总结本次结构优化的核心思路与扩展方式，便于后续复现、对比实验与快速扩展。

## 1. 结构优化要点

- 统一配置入口：`config.py` 使用 `Config` 数据类集中管理超参数。
- 解耦攻击与数据：数据集通过 `backdoor_factory` 注入后门逻辑，避免硬依赖。
- 可插拔注册机制：攻击与模型通过注册表动态选择，减少主流程改动。

## 2. 注册机制（Registry）

注册表定义在 `core/registry.py`：

- `ATTACKS`：注册攻击/后门类
- `MODELS`：注册模型构建函数

示例（注册攻击）：

```python
from core.registry import ATTACKS

@ATTACKS.register("my_attack")
class MyAttack:
    ...
```

示例（注册模型）：

```python
from core.registry import MODELS

@MODELS.register("my_model")
def build_model(num_classes=10):
    ...
```

## 3. 配置与命令行覆盖

`config.py` 提供 `load_config()`，可在运行时覆盖参数，例如：

```bash
python main.py --dataset CIFAR10 --num-clients 20 --freq-strategy ANB
```

关键字段说明：

- `backdoor_name`：选择攻击注册名
- `model_name`：选择模型注册名
- `results_dir` / `weights_dir`：结果与权重输出目录

## 4. 数据与后门解耦方式

以下数据集类新增 `backdoor_factory` 参数：

- `BackdoorDataset`
- `PoisonedTestDataset`
- `MultiTriggerTestDataset`

这样可以在不修改数据代码的情况下切换后门逻辑，实现快速对比实验。

## 5. 建议的扩展流程

1. 新增攻击类/模型并注册到 `ATTACKS` 或 `MODELS`
2. 在运行时通过 CLI 或配置切换名称
3. 保持主流程 `main.py` 不变，实现快速可插拔
