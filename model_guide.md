# model_guide

本文档用于说明本项目在本地与 Modal 上的实验运行方式、结果产物位置，以及当前基线状态。

## 1. 快速运行

### 1.1 本地运行（主入口）

```bash
python main.py
```

### 1.2 Modal 冒烟测试（推荐先跑）

```bash
modal run modal_train.py::run_single --condition anb_freqfed --num-rounds 10
```

### 1.3 Modal 单条件正式实验（示例）

```bash
modal run modal_train.py::run_single --condition anb_freqfed --num-rounds 50
```

## 2. 实验结果放置位置（重点）

### 2.1 本地运行结果位置

当通过 `main.py` 运行时，默认保存到：

- `./results/`
- `./results/weights/`

典型文件：

- `./results/model_<experiment_name>.pth`
- `./results/history_<experiment_name>.json`
- `./results/weights/client_weights_round_5.pkl`
- `./results/weights/client_weights_round_10.pkl`（或最终轮）

### 2.2 Modal 云端结果位置

当通过 `modal_train.py` 运行时，结果保存到 Modal Volume `safb-results` 下的 `/results` 目录。

以 `anb_freqfed` 为例：

- `/results/anb_freqfed/history.json`
- `/results/anb_freqfed/model_final.pth`
- `/results/anb_freqfed/weights/client_weights_round_5.pkl`
- `/results/anb_freqfed/weights/client_weights_round_10.pkl`

你本次日志中的模型路径就是：

- `/results/anb_freqfed/model_final.pth`

### 2.3 将 Modal 结果拉回本地

```bash
modal volume get safb-results /results ./results_from_modal
```

拉取后对应目录：

- `./results_from_modal/anb_freqfed/history.json`
- `./results_from_modal/anb_freqfed/model_final.pth`
- `./results_from_modal/anb_freqfed/weights/`

## 3. 当前 10 轮冒烟基线（2026-03-02）

基于你提供的日志，当前 `anb_freqfed` 在 10 轮下的最终表现为：

- `ACC = 10.00%`
- `ASR (Single) = 100.00%`
- `ASR (Multi) = 100.00%`
- `Bypass = 100.00%`

现象：

- 仅第 2 轮出现短暂抑制（ASR=0，Bypass=0）。
- 其余轮次几乎全部恢复到高 ASR 与高绕过。
- 清洁精度长期停留在 10%，说明当前训练/聚合在该配置下效果异常或未收敛。

## 4. 改进跟踪文档

改进计划与实时进展记录放在：

- `Improvement Plan.md`

每次实验后请更新：

- 运行配置（客户端数、恶意比例、轮次、防御参数）
- 关键指标（ACC / ASR / Bypass / Recall / Precision / F1）
- 对应结果文件路径（`history.json`、`model_final.pth`、`weights/*.pkl`）

