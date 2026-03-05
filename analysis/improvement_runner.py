from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT_ROOT / "results" / "improvement_runs"
RUNS_DIR = RESULT_ROOT / "runs"
PULL_CACHE_DIR = RESULT_ROOT / "_pull_cache"
INDEX_JSONL = RESULT_ROOT / "index.jsonl"
SUMMARY_JSON = RESULT_ROOT / "summary_by_group.json"
SUMMARY_MD = RESULT_ROOT / "summary_by_group.md"


ROUND_OPTIONS = (30, 50)
SEED_OPTIONS = (42, 123, 2026)


EXPERIMENT_GROUPS: list[dict[str, Any]] = [
    # === 方案二 + 参数调整组合实验 ===
    # 核心思路：降低 poison_rate 保留主任务能力 + 后门增强损失 + 调整 scaling_factor
    {
        "group": "solution_v1",
        "condition": "anb_freqfed",
        "poison_rate": 0.75,
        "scaling_factor": 5.5,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "backdoor_boost_weight": 0.3,
    },
    {
        "group": "solution_v2",
        "condition": "anb_freqfed",
        "poison_rate": 0.80,
        "scaling_factor": 5.0,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "backdoor_boost_weight": 0.3,
    },
    {
        "group": "solution_v3",
        "condition": "anb_freqfed",
        "poison_rate": 0.70,
        "scaling_factor": 6.0,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "backdoor_boost_weight": 0.3,
    },
    # === 强后门增强版本（更高的 boost weight）===
    {
        "group": "strong_boost_v1",
        "condition": "anb_freqfed",
        "poison_rate": 0.75,
        "scaling_factor": 5.0,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "backdoor_boost_weight": 0.5,
    },
    {
        "group": "strong_boost_v2",
        "condition": "anb_freqfed",
        "poison_rate": 0.80,
        "scaling_factor": 4.5,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "backdoor_boost_weight": 0.5,
    },
]


def _build_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for group_cfg in EXPERIMENT_GROUPS:
        for rounds in ROUND_OPTIONS:
            for seed in SEED_OPTIONS:
                name = f"{group_cfg['group']}_r{rounds}_s{seed}"
                experiments.append(
                    {
                        "name": name,
                        "group": group_cfg["group"],
                        "condition": group_cfg["condition"],
                        "num_rounds": rounds,
                        "poison_rate": group_cfg["poison_rate"],
                        "scaling_factor": group_cfg["scaling_factor"],
                        "local_epochs": group_cfg["local_epochs"],
                        "learning_rate": group_cfg["learning_rate"],
                        "backdoor_boost_weight": group_cfg.get("backdoor_boost_weight", 0.3),
                        "seed": seed,
                    }
                )
    return experiments


EXPERIMENTS = _build_experiments()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _last(values: list[Any] | None) -> Any:
    if not values:
        return None
    return values[-1]


def _normalize_ratio(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric / 100.0 if numeric > 1.0 else numeric


def _fmt_pct(value: Any) -> str:
    ratio = _normalize_ratio(value)
    if ratio is None:
        return "-"
    return f"{ratio:.2%}"


def _extract_metrics(history_path: Path) -> dict[str, Any]:
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    return {
        "acc_final": _last(history.get("test_acc")),
        "asr_single_final": _last(history.get("test_asr")),
        "asr_multi_final": _last(history.get("test_asr_multi")),
        "bypass_final": _last(history.get("defense_bypass_rate")),
        "recall_final": _last(history.get("defense_recall")),
        "precision_final": _last(history.get("defense_precision")),
        "f1_final": _last(history.get("defense_f1")),
        "rounds_recorded": len(history.get("test_acc", [])),
    }


def _extract_metrics_from_run_log(run_log_path: Path) -> dict[str, Any]:
    if not run_log_path.exists():
        return {}

    text = run_log_path.read_text(encoding="utf-8", errors="replace")

    def _first_ratio(pattern: str) -> float | None:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(1)) / 100.0

    def _last_ratio(pattern: str) -> float | None:
        matches = re.findall(pattern, text)
        if not matches:
            return None
        return float(matches[-1]) / 100.0

    metrics = {
        "acc_final": _first_ratio(r"Final Test Accuracy:\s*([0-9.]+)%"),
        "asr_single_final": _first_ratio(r"Final ASR \(Single Trigger\):\s*([0-9.]+)%"),
        "asr_multi_final": _first_ratio(r"Final ASR \(Multi-Trigger\):\s*([0-9.]+)%"),
        "bypass_final": _last_ratio(r"Bypass Rate \(evaded/malicious\):\s*([0-9.]+)%"),
        "recall_final": _last_ratio(r"Recall\s*\(detected/malicious\):\s*([0-9.]+)%"),
        "precision_final": _last_ratio(r"Precision \(true/flagged\):\s*([0-9.]+)%"),
        "f1_final": _last_ratio(r"F1 Score:\s*([0-9.]+)%"),
    }

    return {key: value for key, value in metrics.items() if value is not None}


def _run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.pop("SSLKEYLOGFILE", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    with log_path.open("w", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            logfile.write(line)

        return process.wait()


def _find_history_file(search_root: Path) -> Path | None:
    candidates = list(search_root.rglob("history.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0]


def _append_index(record: dict[str, Any]) -> None:
    with INDEX_JSONL.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_index_records() -> list[dict[str, Any]]:
    if not INDEX_JSONL.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in INDEX_JSONL.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean_v = statistics.mean(values)
    std_v = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v


def _fmt_mean_std(mean_v: float | None, std_v: float | None) -> str:
    if mean_v is None or std_v is None:
        return "-"
    return f"{mean_v:.2%} ± {std_v:.2%}"


def summarize_index(write_files: bool = True) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for record in _read_index_records():
        metrics = record.get("metrics") or {}
        exp = record.get("experiment") or {}
        if not metrics:
            continue

        condition = exp.get("condition")
        num_rounds = exp.get("num_rounds")
        poison_rate = exp.get("poison_rate")
        scaling_factor = exp.get("scaling_factor")
        local_epochs = exp.get("local_epochs")
        learning_rate = exp.get("learning_rate")
        backdoor_boost_weight = exp.get("backdoor_boost_weight", 0.3)

        if condition is None or num_rounds is None:
            continue

        group_name = exp.get("group")
        if not group_name:
            group_name = (
                f"legacy_pr{poison_rate}_sf{scaling_factor}_"
                f"ep{local_epochs}_lr{learning_rate}"
            )

        key = (
            group_name,
            condition,
            num_rounds,
            poison_rate,
            scaling_factor,
            local_epochs,
            learning_rate,
            backdoor_boost_weight,
        )

        grouped[key].append(
            {
                "seed": exp.get("seed"),
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "metrics_source": record.get("metrics_source"),
                "metrics": metrics,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for key, entries in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        (
            group_name,
            condition,
            num_rounds,
            poison_rate,
            scaling_factor,
            local_epochs,
            learning_rate,
        ) = key

        acc_vals = [_normalize_ratio(e["metrics"].get("acc_final")) for e in entries]
        asr_vals = [_normalize_ratio(e["metrics"].get("asr_single_final")) for e in entries]
        bypass_vals = [_normalize_ratio(e["metrics"].get("bypass_final")) for e in entries]

        acc_vals = [v for v in acc_vals if v is not None]
        asr_vals = [v for v in asr_vals if v is not None]
        bypass_vals = [v for v in bypass_vals if v is not None]

        acc_mean, acc_std = _mean_std(acc_vals)
        asr_mean, asr_std = _mean_std(asr_vals)
        bypass_mean, bypass_std = _mean_std(bypass_vals)

        pass_count = 0
        for entry in entries:
            asr = _normalize_ratio(entry["metrics"].get("asr_single_final"))
            bypass = _normalize_ratio(entry["metrics"].get("bypass_final"))
            if asr is not None and bypass is not None and asr >= 0.85 and bypass >= 0.70:
                pass_count += 1

        unique_seeds = sorted({seed for seed in (e.get("seed") for e in entries) if seed is not None})

        summary_rows.append(
            {
                "group": group_name,
                "condition": condition,
                "num_rounds": num_rounds,
                "poison_rate": poison_rate,
                "scaling_factor": scaling_factor,
                "local_epochs": local_epochs,
                "learning_rate": learning_rate,
                "backdoor_boost_weight": backdoor_boost_weight,
                "num_runs": len(entries),
                "seeds": unique_seeds,
                "seed_coverage": f"{len(unique_seeds)}/{len(SEED_OPTIONS)}",
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "asr_mean": asr_mean,
                "asr_std": asr_std,
                "bypass_mean": bypass_mean,
                "bypass_std": bypass_std,
                "constraint_pass_count": pass_count,
            }
        )

    if write_files:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "constraint": {"min_asr": 0.85, "min_bypass": 0.70},
            "rows": summary_rows,
        }
        SUMMARY_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        md_lines = [
            "# Improvement Summary by Group",
            "",
            f"Generated at: {payload['generated_at']}",
            "",
            "Constraint: `ASR >= 85%` and `Bypass >= 70%`",
            "",
            "| Group | Rounds | poison_rate | scaling_factor | boost_weight | local_epochs | lr | seeds | ACC (mean±std) | ASR (mean±std) | Bypass (mean±std) | pass_count |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|",
        ]

        for row in summary_rows:
            md_lines.append(
                "| "
                f"{row['group']} | {row['num_rounds']} | "
                f"{row['poison_rate']:.2f} | {row['scaling_factor']:.1f} | "
                f"{row['backdoor_boost_weight']:.1f} | "
                f"{row['local_epochs']} | {row['learning_rate']:.3f} | "
                f"{row['seed_coverage']} ({','.join(map(str, row['seeds']))}) | "
                f"{_fmt_mean_std(row['acc_mean'], row['acc_std'])} | "
                f"{_fmt_mean_std(row['asr_mean'], row['asr_std'])} | "
                f"{_fmt_mean_std(row['bypass_mean'], row['bypass_std'])} | "
                f"{row['constraint_pass_count']} |"
            )

        SUMMARY_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return summary_rows


def run_one(exp: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    run_id = f"{_timestamp()}_{exp['name']}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_log = run_dir / "run.log"
    meta_path = run_dir / "meta.json"
    pull_dir = PULL_CACHE_DIR / run_id
    artifacts_dir = run_dir / "artifacts"

    modal_run_cmd = [
        "modal",
        "run",
        "modal_train.py::run_single",
        "--condition",
        str(exp["condition"]),
        "--num-rounds",
        str(exp["num_rounds"]),
        "--poison-rate",
        str(exp["poison_rate"]),
        "--scaling-factor",
        str(exp["scaling_factor"]),
        "--local-epochs",
        str(exp["local_epochs"]),
        "--learning-rate",
        str(exp["learning_rate"]),
        "--backdoor-boost-weight",
        str(exp["backdoor_boost_weight"]),
        "--seed",
        str(exp["seed"]),
    ]

    modal_get_cmd = [
        "modal",
        "volume",
        "get",
        "safb-results",
        f"/results/{exp['condition']}",
        str(pull_dir),
    ]

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    record: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "experiment": exp,
        "commands": {
            "modal_run": " ".join(modal_run_cmd),
            "modal_get": " ".join(modal_get_cmd),
        },
        "paths": {
            "run_dir": str(run_dir),
            "run_log": str(run_log),
            "meta": str(meta_path),
            "pull_dir": str(pull_dir),
            "artifacts": str(artifacts_dir),
        },
        "status": "pending",
        "metrics": {},
    }

    if dry_run:
        record["status"] = "dry_run"
        record["ended_at"] = datetime.now().isoformat(timespec="seconds")
        record["duration_sec"] = round(time.time() - start_time, 2)
        meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_index(record)
        return record

    run_code = _run_command(modal_run_cmd, run_log)
    record["run_exit_code"] = run_code
    if run_code != 0:
        record["status"] = "run_failed"
        record["metrics"] = _extract_metrics_from_run_log(run_log)
        if record["metrics"]:
            record["metrics_source"] = "run_log"
        record["ended_at"] = datetime.now().isoformat(timespec="seconds")
        record["duration_sec"] = round(time.time() - start_time, 2)
        meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_index(record)
        return record

    if pull_dir.exists():
        shutil.rmtree(pull_dir)
    pull_dir.parent.mkdir(parents=True, exist_ok=True)
    pull_code = _run_command(modal_get_cmd, run_dir / "pull.log")
    record["pull_exit_code"] = pull_code
    if pull_code != 0:
        record["status"] = "pull_failed"
        record["metrics"] = _extract_metrics_from_run_log(run_log)
        if record["metrics"]:
            record["metrics_source"] = "run_log"
        record["ended_at"] = datetime.now().isoformat(timespec="seconds")
        record["duration_sec"] = round(time.time() - start_time, 2)
        meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_index(record)
        return record

    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    shutil.copytree(pull_dir, artifacts_dir)

    history_file = _find_history_file(artifacts_dir)
    if history_file is not None:
        record["metrics"] = _extract_metrics(history_file)
        record["metrics_source"] = "history"
        record["history_file"] = str(history_file)
        record["status"] = "ok"
    else:
        record["metrics"] = _extract_metrics_from_run_log(run_log)
        if record["metrics"]:
            record["metrics_source"] = "run_log"
        record["status"] = "ok_no_history"

    record["ended_at"] = datetime.now().isoformat(timespec="seconds")
    record["duration_sec"] = round(time.time() - start_time, 2)

    meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_index(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Improvement experiment runner")
    parser.add_argument("--only", type=str, default=None, help="Run only one experiment name")
    parser.add_argument("--group", type=str, default=None, help="Run only one group (e.g. strong_g1)")
    parser.add_argument("--rounds", type=int, choices=ROUND_OPTIONS, default=None, help="Filter by rounds")
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed")
    parser.add_argument("--dry-run", action="store_true", help="Only generate run metadata")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop when one run fails")
    parser.add_argument("--list", action="store_true", help="List all generated experiments")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize index.jsonl")
    args = parser.parse_args()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("Available experiments:")
        for exp in EXPERIMENTS:
            print(
                f"  {exp['name']}: group={exp['group']}, rounds={exp['num_rounds']}, "
                f"seed={exp['seed']}, poison_rate={exp['poison_rate']}, "
                f"scaling={exp['scaling_factor']}, epochs={exp['local_epochs']}, lr={exp['learning_rate']}"
            )
        return

    if args.summarize_only:
        rows = summarize_index(write_files=True)
        print(f"Summary rows: {len(rows)}")
        print(f"  JSON: {SUMMARY_JSON}")
        print(f"  MD  : {SUMMARY_MD}")
        return

    selected = EXPERIMENTS
    if args.only:
        selected = [item for item in selected if item["name"] == args.only]
        if not selected:
            raise ValueError(f"No experiment named '{args.only}'")
    if args.group:
        selected = [item for item in selected if item["group"] == args.group]
    if args.rounds is not None:
        selected = [item for item in selected if item["num_rounds"] == args.rounds]
    if args.seed is not None:
        selected = [item for item in selected if item["seed"] == args.seed]

    if not selected:
        raise ValueError("No experiments matched the provided filters")

    print("=" * 80)
    print("Improvement Experiment Runner")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Result root  : {RESULT_ROOT}")
    print(f"Total runs   : {len(selected)}")
    print("=" * 80)

    for index, exp in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] Running: {exp['name']}")
        print(
            "  params => "
            f"group={exp['group']}, condition={exp['condition']}, rounds={exp['num_rounds']}, "
            f"seed={exp['seed']}, poison_rate={exp['poison_rate']}, "
            f"scaling_factor={exp['scaling_factor']}, local_epochs={exp['local_epochs']}, "
            f"lr={exp['learning_rate']}"
        )

        result = run_one(exp, dry_run=args.dry_run)
        print(f"  status => {result['status']}")

        metrics = result.get("metrics", {})
        if metrics:
            print(
                "  final  => "
                f"ACC={_fmt_pct(metrics.get('acc_final'))}, "
                f"ASR={_fmt_pct(metrics.get('asr_single_final'))}, "
                f"Bypass={_fmt_pct(metrics.get('bypass_final'))}"
            )

        if args.stop_on_error and result["status"] not in {"ok", "ok_no_history", "dry_run", "pull_failed"}:
            print("Stopped due to error status.")
            break

    rows = summarize_index(write_files=True)
    print("\nDone.")
    print(f"  Index   : {INDEX_JSONL}")
    print(f"  Summary : {SUMMARY_MD} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

