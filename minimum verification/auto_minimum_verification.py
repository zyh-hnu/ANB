from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.2%}"


def _run_scenario(project_root: Path, args, rounds: int) -> dict[str, Any]:
    script = project_root / "minimum verification" / "run_minimum_verification.py"
    out_dir = Path(args.output_root) / f"r{rounds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        args.dataset,
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(out_dir),
        "--num-rounds",
        str(rounds),
        "--num-clients",
        str(args.num_clients),
        "--poison-ratio",
        str(args.poison_ratio),
        "--poison-rate",
        str(args.poison_rate),
        "--scaling-factor",
        str(args.scaling_factor),
        "--local-epochs",
        str(args.local_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--train-subset",
        str(args.train_subset),
        "--test-subset",
        str(args.test_subset),
        "--defense-method",
        args.defense_method,
        "--conditions",
        *args.conditions,
        "--seeds",
        *[str(s) for s in args.seeds],
    ]

    print("\n" + "=" * 90)
    print("Running scenario:", f"rounds={rounds}")
    print("Command:", " ".join(cmd))
    print("=" * 90)

    subprocess.check_call(cmd, cwd=project_root)

    summary_path = out_dir / "minimum_verification_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "rounds": rounds,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "summary": payload.get("summary", {}),
    }


def _write_combined_report(output_root: Path, all_payloads: list[dict[str, Any]]) -> None:
    combined_json = output_root / "combined_summary.json"
    combined_md = output_root / "combined_summary.md"

    combined_json.write_text(
        json.dumps({"scenarios": all_payloads}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Combined Minimum Verification Summary",
        "",
        "| rounds | asr_gain (ANB-FIXED) | bypass_gain (ANB-FIXED) | supported |",
        "|---:|---:|---:|---|",
    ]
    for item in all_payloads:
        hypothesis = item.get("summary", {}).get("hypothesis_check", {})
        lines.append(
            f"| {item['rounds']} | {_fmt_pct(hypothesis.get('asr_gain'))} | "
            f"{_fmt_pct(hypothesis.get('bypass_gain'))} | {hypothesis.get('supported')} |"
        )

    combined_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nCombined reports saved:")
    print("-", combined_json)
    print("-", combined_md)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated runner for minimum verification (multi-round scenarios)",
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./results/minimum_verification_auto")

    parser.add_argument("--rounds", nargs="+", type=int, default=[30, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["fixed_freqfed", "anb_freqfed"],
        choices=["fixed_freqfed", "anb_freqfed", "anb_no_defense"],
    )

    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--poison-ratio", type=float, default=0.2)
    parser.add_argument("--poison-rate", type=float, default=0.9)
    parser.add_argument("--scaling-factor", type=float, default=4.5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--train-subset", type=int, default=6000)
    parser.add_argument("--test-subset", type=int, default=1200)
    parser.add_argument("--defense-method", type=str, default="hdbscan")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_payloads = []
    for rounds in args.rounds:
        payload = _run_scenario(project_root, args, rounds)
        all_payloads.append(payload)

    _write_combined_report(output_root, all_payloads)


if __name__ == "__main__":
    main()

