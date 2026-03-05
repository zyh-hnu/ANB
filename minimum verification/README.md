# Minimum Verification（最小化验证）

这个目录只做一件事：

- 以**最小实验集合**验证你的核心方法是否成立（ANB 相对 FIXED 在防御下是否更有效）。

不包含论文全套流程、不包含大规模可视化、不包含其余扩展实验。

---

## 1. 目录内容

- `run_minimum_verification.py`：最小验证主脚本（核心）。
- `kaggle_minimum_verification.ipynb`：Kaggle 一键运行 Notebook。
- `auto_minimum_verification.py`：自动化批量运行脚本（30/50轮等场景）。

---

## 2. 在 Kaggle 需要上传什么文件？

### 推荐方式（最稳）

直接上传**整个项目目录 SAFB**（或打包为 zip 后上传再解压）。

原因：`run_minimum_verification.py` 会复用项目里的模块（`core/`、`data/`、`federated/`、`models/`、`main.py`、`config.py` 等）。

### 如果你只想上传最小必要文件

至少要包含：

- `minimum verification/`（本目录全部文件）
- `config.py`
- `main.py`
- `requirements.txt`
- `core/`
- `data/`
- `federated/`
- `models/`

---

## 3. Kaggle 一键运行方式

### 方式 A：直接打开 Notebook

1. 在 Kaggle 新建 Notebook（建议开启 GPU）。
2. 上传/挂载项目文件。
3. 打开并运行：
   - `minimum verification/kaggle_minimum_verification.ipynb`

这个 notebook 会自动：

- 定位项目根目录
- 安装依赖
- 执行最小验证（默认 `fixed_freqfed` vs `anb_freqfed`，3 seeds）
- 打印 summary 与 hypothesis check

> 注意：Kaggle 的 `/kaggle/input` 是只读目录，结果文件必须写到 `/kaggle/working`。

---

## 4. 命令行运行（Kaggle 或本地）

### 4.1 最小快速验证（30轮，1 seed）

```bash
python "minimum verification/run_minimum_verification.py" \
  --output-dir /kaggle/working/results/minimum_verification \
  --num-rounds 30 \
  --train-subset 6000 \
  --test-subset 1200 \
  --seeds 42
```

### 4.2 更稳验证（30轮，3 seeds）

```bash
python "minimum verification/run_minimum_verification.py" \
  --output-dir /kaggle/working/results/minimum_verification \
  --num-rounds 30 \
  --train-subset 8000 \
  --test-subset 2000 \
  --poison-rate 0.9 \
  --scaling-factor 4.5 \
  --local-epochs 3 \
  --seeds 42 123 2026
```

### 4.3 自动化多场景（30 + 50轮，一条命令）

```bash
python "minimum verification/auto_minimum_verification.py" \
  --output-root /kaggle/working/results/minimum_verification_auto \
  --rounds 30 50 \
  --seeds 42 123 2026 \
  --poison-rate 0.9 \
  --scaling-factor 4.5
```

---

## 5. 输出文件

### 单次最小验证

- `./results/minimum_verification/minimum_verification_summary.json`

### 自动化批量验证

- `./results/minimum_verification_auto/r30/minimum_verification_summary.json`
- `./results/minimum_verification_auto/r50/minimum_verification_summary.json`
- `./results/minimum_verification_auto/combined_summary.json`
- `./results/minimum_verification_auto/combined_summary.md`

---

## 6. 如何判断“方法是否正确”

看 summary 里的两项：

- `ASR gain (ANB - FIXED)` 是否 > 0
- `Bypass gain (ANB - FIXED)` 是否 >= 0（或至少不下降）

JSON 字段：

- `summary.hypothesis_check.supported`

`true` 表示在当前最小验证设定下支持你的核心方法假设。

---

## 7. 自动化与 Token 说明

### Q1：能不能自动化运行最小化验证？

可以。

- 本地/服务器自动化：直接用 `auto_minimum_verification.py`（不需要 Kaggle token）。
- Kaggle Notebook 内自动化：打开 `kaggle_minimum_verification.ipynb` 一键全部运行（不需要 token）。

### Q2：什么时候需要 Kaggle Token？

仅当你要在**本地机器通过 Kaggle API 自动提交/更新 Kaggle Notebook**时才需要。

你需要：

1. 在 Kaggle 账号页面创建 API Token（下载 `kaggle.json`）
2. 本地放到 `~/.kaggle/kaggle.json`（Linux/macOS）或 `%USERPROFILE%\.kaggle\kaggle.json`（Windows）
3. 安装 `kaggle` CLI 后用 `kaggle kernels push` 自动提交

如果你后续要这条链路，我可以再给你补一个 `kaggle kernels push` 的完整脚本模板。
