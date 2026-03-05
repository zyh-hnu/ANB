# improvement_runs

用于保存改进阶段的实验结果。

- 每次批量运行都会在 `runs/` 下生成唯一目录（时间戳 + 参数名）。
- 每个运行目录包含：
  - `run.log`：`modal run` 输出
  - `pull.log`：`modal volume get` 输出
  - `meta.json`：参数、状态、关键指标、路径
  - `artifacts/`：拉取下来的实验文件（含 `history.json`、`model_final.pth`、`weights/`）
- 全局索引：`index.jsonl`

执行脚本：

```bash
python analysis/improvement_runner.py
```

