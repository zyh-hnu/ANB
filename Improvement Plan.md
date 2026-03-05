# Improvement Plan

> 研究目标调整：当前**Bypass 已经足够好**，后续工作以“**保持攻击成功（ASR/Bypass高）并显著提升 Clean ACC**”为主。

## 1. 当前基线（已确认）

更新时间：2026-03-02

- 任务：`anb_freqfed`（ANB 攻击 + hdbscan 防御）
- 规模：10 clients（恶意2 / 良性8），10 rounds
- 当前结果：
  - `ACC = 10.00%`
  - `ASR(single) = 100.00%`
  - `ASR(multi) = 100.00%`
  - `Bypass = 100.00%`

结论：

- 攻击端目标（ASR/Bypass）已达成。
- 主要短板是主任务可用性（Clean ACC 过低）。

---

## 2. 从代码出发的关键判断（已阅读项目）

### 2.1 目前最可能压低 ACC 的因素

- `federated/client.py` 中恶意客户端默认 `poison_rate=1.0`（非目标类样本几乎全毒化）。
- 恶意更新使用 `scaling_factor=5.0`（模型替换放大），对全局主任务破坏强。
- 当前 `local_epochs=5` + `lr=0.01` 在强攻击设置下，可能进一步放大恶意漂移。
- 10 轮冒烟偏短，但“ACC 长期 10%”已说明当前参数组合失衡。

### 2.2 研究策略

- 不再追求更高 Bypass（已 100%）。
- 转为优化“攻击-可用性平衡点”：
  - 约束：`ASR >= 85%`，`Bypass >= 70%`
  - 目标：优先将 `ACC` 拉到 `>= 40%`（阶段1），再冲 `>= 60%`（阶段2）

---

## 3. 改进路线（只围绕 ACC）

### Phase A：先把实验调参能力补齐（必要工程）

- A1. 将恶意 `poison_rate` 暴露为可配置参数（当前硬编码 1.0）。
- A2. 支持在命令行/Modal 入口传参：`poison_rate`、`scaling_factor`、`local_epochs`、`lr`。
- A3. 在日志中固定打印“本轮有效攻击参数快照”，保证可复现与可比较。

完成标准：可以一键进行参数扫面，不改代码即可复现实验。

### Phase B：ACC 优先调参（保持攻击有效）

- B1. 先扫 `scaling_factor`：`[5.0, 4.0, 3.0, 2.5, 2.0]`
- B2. 再扫 `poison_rate`：`[1.0, 0.7, 0.5, 0.3, 0.2]`
- B3. 轻量训练参数扫面：
  - `local_epochs`: `[5, 3, 2]`
  - `learning_rate`: `[0.01, 0.005]`

筛选规则（硬约束）：

- 若 `ASR < 85%` 或 `Bypass < 70%`，该组淘汰。
- 在剩余组中按 `ACC` 排序，选 Top-3 进入下一阶段。

### Phase C：稳定性与论文可用结果

- C1. Top-3 配置做 3 个随机种子复验（seed: 42/123/2026）。
- C2. 轮次扩展验证：`10 -> 20 -> 50`，观察 ACC/ASR/Bypass 曲线是否稳定。
- C3. 与基线配置（当前 10% ACC 版本）做对照表，形成论文主表。

---

## 4. 实时跟踪看板

状态：`TODO / DOING / DONE / BLOCKED`

| ID | 任务 | 产出 | 状态 | 最近更新 |
|---|---|---|---|---|
| A1 | 暴露 `poison_rate` 配置项 | 可从配置控制毒化比例 | DONE | 2026-03-03 |
| A2 | 扩展 CLI/Modal 传参 | 免改代码批量调参 | DONE | 2026-03-03 |
| A3 | 增加参数快照日志 | 每次结果可追溯 | DONE | 2026-03-03 |
| B1 | 扫 `scaling_factor` | ACC/ASR/Bypass 对照表 | DOING | 2026-03-03（已测 3.0 / 2.5） |
| B2 | 扫 `poison_rate` | 平衡点候选配置 | DOING | 2026-03-03（已测 0.5 / 0.3） |
| B3 | 扫训练超参 | 进一步提升 ACC | TODO | - |
| C1 | 多 seed 复验 | 稳定性结论 | TODO | - |
| C2 | 扩展到 20/50 轮 | 论文曲线数据 | TODO | - |
| C3 | 形成论文结果表 | 可直接入文稿 | TODO | - |

---

## 5. 记录模板（每次实验后追加）

```md
### [YYYY-MM-DD HH:mm] 实验记录
- Goal: 提升 ACC（保持攻击）
- Command:
- Key Params: scaling_factor= , poison_rate= , local_epochs= , lr= , rounds=
- Result: ACC= , ASR(single)= , ASR(multi)= , Bypass= , Recall= , Precision= , F1=
- Output Files:
  - history:
  - model:
  - weights:
- Decision: 保留/淘汰（原因）
- Next:
```

---

## 6. 首条任务（下一步立即执行）

- 优先顺序：`A1 -> A2 -> B1`
- 先不动攻击核心机制（ANB 四组件不删），先通过降低攻击“强度参数”换取 ACC 提升。
- 目标是找出第一组满足：`ACC >= 40%` 且 `ASR >= 85%` 且 `Bypass >= 70%` 的配置。

---

## 7. 结果文件位置（追踪约定）

本地实验：

- `./results/history_<experiment>.json`
- `./results/model_<experiment>.pth`
- `./results/weights/client_weights_round_*.pkl`

Modal 实验：

- `/results/<condition>/history.json`
- `/results/<condition>/model_final.pth`
- `/results/<condition>/weights/client_weights_round_*.pkl`

拉取命令：

```bash
modal volume get safb-results /results ./results_from_modal
```

---

## 8. 结果追加（2026-03-03，来自 `improvement_runner.py`）

数据来源：

- 目录：`./results/improvement_runs/runs/*/run.log`
- 说明：三次运行 `modal run` 均成功，但 `modal volume get` 失败（`pull_failed`），本节指标来自 `run.log` 末尾汇总。

### 8.1 三组参数与最终指标

| Run ID | poison_rate | scaling_factor | local_epochs | lr | ACC | ASR(single) | ASR(multi) | Bypass | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260303_001522_acc_focus_v1 | 0.50 | 3.0 | 3 | 0.005 | 79.52% | 3.63% | 3.71% | 100.00% | 100.00% | 20.00% | 33.33% |
| 20260303_012427_acc_focus_v2 | 0.30 | 3.0 | 3 | 0.005 | 78.81% | 4.86% | 4.93% | 100.00% | 100.00% | 20.00% | 33.33% |
| 20260303_035303_acc_focus_v3 | 0.50 | 2.5 | 3 | 0.005 | 78.32% | 4.80% | 4.92% | 100.00% | 100.00% | 20.00% | 33.33% |

### 8.2 结论（相对 2026-03-02 基线）

- **ACC 显著提升**：`10.00% -> 78%~80%`，主任务可用性已恢复。
- **ASR 大幅下降**：`100% -> <5%`，远低于约束 `ASR >= 85%`，三组均淘汰。
- **Bypass 保持 100%**：防御仍全部放行（Precision 仅 20%），当前瓶颈不是“能否绕过防御”，而是“后门是否学到”。
- **方向判断**：前一轮降强度（`poison_rate/scaling_factor/local_epochs/lr` 同时下调）过度，导致攻击有效性塌陷。

### 8.3 决策与下一轮搜索（继续优化）

下一轮从“恢复 ASR”为主、同时监控 ACC，参数向基线回调但不一步回满：

- 固定：`condition=anb_freqfed`，`num_rounds=10`
- 候选组（建议按顺序跑）：
  1. `poison_rate=0.60, scaling_factor=3.5, local_epochs=4, lr=0.01`
  2. `poison_rate=0.70, scaling_factor=4.0, local_epochs=4, lr=0.01`
  3. `poison_rate=0.80, scaling_factor=4.0, local_epochs=4, lr=0.01`
  4. `poison_rate=0.80, scaling_factor=4.5, local_epochs=5, lr=0.01`
  5. `poison_rate=1.00, scaling_factor=4.0, local_epochs=4, lr=0.01`

筛选规则（保持不变）：

- 先过硬约束：`ASR >= 85%` 且 `Bypass >= 70%`
- 再按 `ACC` 排序，取 Top-3 进入 `C1` 多 seed 复验。

执行备注：

- 若继续出现 `pull_failed`，本地先以 `run.log` 为准做决策；后续需修复 `modal volume get` 路径问题，保证 `history.json` 可拉取和归档。

---

## 9. 结果追加（2026-03-04 ~ 2026-03-05，ASR Recovery 系列）

数据来源：

- 目录：`./results/improvement_runs/runs/*/run.log`
- 说明：五次运行 `modal run` 均成功，`modal volume get` 失败（`pull_failed`），指标来自 `run.log` 末尾汇总。

### 9.1 五组参数与最终指标

| Run ID | poison_rate | scaling_factor | local_epochs | lr | ACC | ASR(single) | ASR(multi) | Bypass | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260304_181215_asr_recovery_v1 | 0.60 | 3.5 | 4 | 0.01 | 76.31% | 3.79% | 4.08% | 100.00% | 淘汰 |
| 20260304_194350_asr_recovery_v2 | 0.70 | 4.0 | 4 | 0.01 | 73.76% | 4.69% | 5.03% | 100.00% | 淘汰 |
| 20260304_210804_asr_recovery_v3 | 0.80 | 4.0 | 4 | 0.01 | 68.54% | 19.40% | 19.96% | 100.00% | 淘汰 |
| 20260304_223255_asr_recovery_v4 | 0.80 | 4.5 | 5 | 0.01 | 70.53% | 17.21% | 19.16% | 100.00% | 淘汰 |
| 20260305_001116_asr_recovery_v5 | 1.00 | 4.0 | 4 | 0.01 | 10.00% | 100.00% | 100.00% | 100.00% | 淘汰 |

### 9.2 关键发现

**问题定位**：ACC 与 ASR 呈现**零和博弈**关系

| poison_rate | ACC | ASR | 现象 |
|---|---|---|---|
| 0.6 ~ 0.7 | 73%~76% | <5% | 后门被良性更新稀释，学不到 |
| 0.8 | 68%~70% | 17%~20% | 后门开始学到，但仍不够 |
| 1.0 | 10% | 100% | 主任务完全崩溃，只学后门 |

**根因分析**：

1. `poison_rate=1.0` 时，恶意客户端的所有非目标类样本都被毒化 → 标签全变 target_label → 模型只学预测一个类
2. 恶意客户端 Loss ≈ 0 说明本地训练只优化后门，不学主任务
3. scaling_factor 放大了恶意更新，但恶意更新本身不包含主任务知识

### 9.3 核心结论

> **当前瓶颈不是参数调整能解决的**，而是恶意客户端训练策略的问题。
> 
> 恶意客户端需要在学习后门的同时保留主任务能力，否则无法同时达成 ACC 和 ASR 目标。

### 9.4 解决方案（2026-03-05 实施）

**方案一：参数调整**
- 降低 `poison_rate`（如 0.75）保留部分良性样本
- 调高 `scaling_factor`（如 5.5）补偿后门信号稀释

**方案二：后门增强损失**（已实现）
- 在恶意客户端训练中添加 `backdoor_boost_loss`
- 对非目标样本，鼓励模型对目标类有更高预测概率
- 让模型同时学习主任务和后门模式

代码修改：
- `federated/client.py`：添加 `backdoor_boost_weight` 参数和损失计算
- `config.py`：添加配置项
- `analysis/improvement_runner.py`：添加新实验组

### 9.5 下一轮实验配置

| Group | poison_rate | scaling_factor | boost_weight | local_epochs |
|---|---:|---:|---:|---:|
| solution_v1 | 0.75 | 5.5 | 0.3 | 3 |
| solution_v2 | 0.80 | 5.0 | 0.3 | 3 |
| solution_v3 | 0.70 | 6.0 | 0.3 | 3 |
| strong_boost_v1 | 0.75 | 5.0 | 0.5 | 3 |
| strong_boost_v2 | 0.80 | 4.5 | 0.5 | 3 |

**预期目标**：
- ACC ≥ 60%
- ASR ≥ 85%
- Bypass ≥ 70%

**运行命令**：
```bash
# 本地快速验证
python main.py --poison-rate 0.75 --scaling-factor 5.5 --backdoor-boost-weight 0.3 --num-rounds 20 --local-epochs 3

# Modal 完整实验
python analysis/improvement_runner.py
```
