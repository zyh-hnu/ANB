"""
Figure Collection & Organization Script  [P4-1]

Collects all thesis-relevant figures from scattered locations, regenerates
any that can be produced without GPU (pure NumPy / matplotlib), and writes
them to a unified directory tree:

    results/paper/
        ch2_intro/          # Motivation / defense failure
        ch3_method/         # Attack mechanism figures
        ch4_experiments/    # Experimental result figures
        ch5_appendix/       # GradCAM, frequency residual, etc.
        FIGURE_INDEX.md     # Chapter → filename → description mapping

All figures are saved at 300 dpi, suitable for direct inclusion in a thesis.

Usage (from project root):
    python analysis/collect_figures.py           # full run
    python analysis/collect_figures.py --index-only  # only refresh FIGURE_INDEX.md
"""

import os
import sys
import shutil
import argparse
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------

PAPER_DIR = './results/paper'

CHAPTER_DIRS = {
    'ch2_intro':       os.path.join(PAPER_DIR, 'ch2_intro'),
    'ch3_method':      os.path.join(PAPER_DIR, 'ch3_method'),
    'ch4_experiments': os.path.join(PAPER_DIR, 'ch4_experiments'),
    'ch5_appendix':    os.path.join(PAPER_DIR, 'ch5_appendix'),
}

for d in CHAPTER_DIRS.values():
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Existing figure copy map
# Source priority: report > experiment > readme  (best quality)
# ---------------------------------------------------------------------------

# (src_path, dst_chapter, dst_filename, description)
COPY_MAP = [
    # ── Chapter 2 Introduction ─────────────────────────────────────────────
    (
        'results/figures/report/defense_evasion_concept2.png',
        'ch2_intro',
        'fig1_defense_evasion_concept.png',
        'Motivation: FreqFed defense failure against ANB (defense evasion concept)',
    ),
    (
        'results/figures/experiment/defense_evasion_concept.png',
        'ch2_intro',
        'fig1_defense_evasion_concept_v1.png',
        'Motivation: defense evasion concept (v1, fallback)',
    ),

    # ── Chapter 3 Method ──────────────────────────────────────────────────
    (
        'results/figures/report/trigger_pipeline2.png',
        'ch3_method',
        'fig2_trigger_pipeline.png',
        'ANB trigger generation pipeline (4-stage flow)',
    ),
    (
        'results/figures/report/multi_client_triggers2.png',
        'ch3_method',
        'fig3_frequency_sharding.png',
        'Frequency Sharding: per-client frequency band allocation',
    ),
    (
        'results/figures/report/frequency_comparison2.png',
        'ch3_method',
        'fig4_frequency_comparison.png',
        'Frequency domain comparison: FIXED vs ANB trigger spectrum',
    ),
    (
        'results/figures/report/energy_comparison.png',
        'ch3_method',
        'fig5_energy_distribution.png',
        'Spectral energy distribution: FIXED vs ANB',
    ),
    (
        'results/figures/report/residual_comparison.png',
        'ch3_method',
        'fig6_residual_comparison.png',
        'Trigger residual comparison (FIXED vs ANB, frequency domain)',
    ),

    # ── Chapter 4 Experiments ─────────────────────────────────────────────
    (
        'results/figures/report/strategy_comparison_purity.png',
        'ch4_experiments',
        'fig7_strategy_comparison_main.png',
        'Main result: ASR/ACC comparison FIXED vs ANB under FreqFed defense',
    ),
    (
        'results/figures/experiment/strategy_comparison.png',
        'ch4_experiments',
        'fig7_strategy_comparison_v1.png',
        'Strategy comparison (v1)',
    ),
    (
        'results/figures/report/clustering_anb.png',
        'ch4_experiments',
        'fig8_clustering_anb.png',
        'Defense clustering result: ANB (evades FreqFed, mixes with benign cluster)',
    ),
    (
        'results/figures/report/clustering_fixed2.png',
        'ch4_experiments',
        'fig9_clustering_fixed.png',
        'Defense clustering result: FIXED (detected by FreqFed, separate cluster)',
    ),
    (
        'results/figures/report/clustering_anb_synthetic.png',
        'ch4_experiments',
        'fig8b_clustering_anb_synthetic.png',
        'Clustering visualization: ANB (synthetic weights)',
    ),
    (
        'results/figures/report/clustering_fixed_synthetic.png',
        'ch4_experiments',
        'fig9b_clustering_fixed_synthetic.png',
        'Clustering visualization: FIXED (synthetic weights)',
    ),
    (
        'results/figures/report/defense_sensitivity2.png',
        'ch4_experiments',
        'fig10_defense_sensitivity.png',
        'Defense sensitivity: ASR vs defense strength / threshold sweep',
    ),
    (
        'results/figures/experiment/trigger_evolution.png',
        'ch4_experiments',
        'fig11_trigger_evolution.png',
        'ANB trigger evolution across training rounds (phased chaos)',
    ),

    # ── Chapter 5 Appendix ────────────────────────────────────────────────
    (
        'results/figures/report/frequency_analysis_client0_anb.png',
        'ch5_appendix',
        'figA1_frequency_analysis_client0.png',
        'Per-client frequency analysis: client 0 ANB trigger spectrum',
    ),
    (
        'results/figures/report/residual_anb.png',
        'ch5_appendix',
        'figA2_residual_anb.png',
        'ANB trigger residual map (spatial domain)',
    ),
    (
        'results/figures/report/residual_fixed.png',
        'ch5_appendix',
        'figA3_residual_fixed.png',
        'FIXED trigger residual map (spatial domain)',
    ),
]

# Sensitivity and CIFAR-100 figures are generated at runtime — listed here
# for the index, with paths that will exist after running the sweeps.
GENERATED_FIGURE_INDEX = [
    (
        'ch4_experiments',
        'results/sensitivity/epsilon_curve.png',
        'fig12_sensitivity_epsilon.png',
        'Parameter sensitivity: ASR/ACC vs trigger strength ε',
    ),
    (
        'ch4_experiments',
        'results/sensitivity/poison_ratio_curve.png',
        'fig13_sensitivity_poison_ratio.png',
        'Parameter sensitivity: ASR/ACC vs malicious client ratio',
    ),
    (
        'ch4_experiments',
        'results/sensitivity/alpha_curve.png',
        'fig14_sensitivity_alpha.png',
        'Parameter sensitivity: ASR/ACC vs Dirichlet α (Non-IID degree)',
    ),
    (
        'ch4_experiments',
        'results/sensitivity/num_rounds_curve.png',
        'fig15_sensitivity_num_rounds.png',
        'Parameter sensitivity: ASR/ACC vs number of training rounds',
    ),
    (
        'ch4_experiments',
        'results/sensitivity/sensitivity_summary.png',
        'fig16_sensitivity_summary.png',
        'Parameter sensitivity summary (all 4 parameters, 2×4 grid)',
    ),
    (
        'ch4_experiments',
        'results/cifar100/cifar100_summary.png',
        'fig17_cifar100_generalization.png',
        'Cross-dataset generalization: CIFAR-100 ASR/ACC/bypass comparison',
    ),
    (
        'ch3_method',
        'results/figures/routing_mechanism.png',
        'fig_routing_mechanism.png',
        'Dual-Domain Routing: variance map → frequency/spatial masks → fused trigger',
    ),
    (
        'ch3_method',
        'results/figures/routing_comparison.png',
        'fig_routing_comparison.png',
        'Dual-Domain Routing: routing behaviour across 4 image texture levels',
    ),
    (
        'ch5_appendix',
        'results/figures/routing_scatter.png',
        'figA4_routing_scatter.png',
        'Dual-Domain Routing scatter: per-pixel frequency vs spatial weight',
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_if_exists(src, dst):
    """Copy src to dst. Returns True if copied, False if src missing."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def _copy_generated_if_exists(src, chapter_key, dst_name):
    """Copy a generated figure into the paper directory if it already exists."""
    dst = os.path.join(CHAPTER_DIRS[chapter_key], dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False


# ---------------------------------------------------------------------------
# Step 1 — Regenerate figures that require no GPU
# ---------------------------------------------------------------------------

def regenerate_no_gpu_figures():
    """
    Call visualization scripts that work purely on NumPy / matplotlib
    (no trained model needed).  Each script saves to its own output path;
    we later copy the results into results/paper/.
    """
    print('\n[P4-1] Step 1: Regenerating no-GPU figures at 300 dpi...')

    # -- Dual-Domain Routing figures (analysis/visualize_dual_routing.py) --
    try:
        from analysis.visualize_dual_routing import (
            visualize_routing_mechanism,
            visualize_routing_comparison,
            visualize_routing_scatter,
        )
        print('  Generating routing_mechanism.png...')
        visualize_routing_mechanism(save_path='./results/figures/routing_mechanism.png', dpi=300)
        print('  Generating routing_comparison.png...')
        visualize_routing_comparison(save_path='./results/figures/routing_comparison.png', dpi=300)
        print('  Generating routing_scatter.png...')
        visualize_routing_scatter(save_path='./results/figures/routing_scatter.png', dpi=300)
    except Exception as e:
        print(f'  [WARN] Routing figures failed: {e}')

    # -- Trigger pipeline & multi-client sharding --
    try:
        from analysis.create_visualizations import (
            visualize_trigger_generation_pipeline,
            visualize_multi_client_triggers,
            visualize_frequency_comparison,
            create_defense_evasion_illustration,
        )
        print('  Generating trigger_pipeline.png...')
        visualize_trigger_generation_pipeline(
            save_path='./results/figures/trigger_pipeline_300.png'
        )
        print('  Generating frequency_sharding.png...')
        visualize_multi_client_triggers(
            num_clients=8,
            save_path='./results/figures/frequency_sharding_300.png'
        )
        print('  Generating frequency_comparison.png...')
        visualize_frequency_comparison(
            save_path='./results/figures/frequency_comparison_300.png'
        )
        print('  Generating defense_evasion_concept.png...')
        create_defense_evasion_illustration(
            save_path='./results/figures/defense_evasion_concept_300.png'
        )
    except Exception as e:
        print(f'  [WARN] create_visualizations figures failed: {e}')

    print('[P4-1] Step 1 complete.\n')


# ---------------------------------------------------------------------------
# Step 2 — Copy existing figures into results/paper/<chapter>/
# ---------------------------------------------------------------------------

def collect_existing_figures():
    """Copy best-available existing figures into the paper directory tree."""
    print('[P4-1] Step 2: Collecting existing figures...')
    copied, missing = 0, 0

    for src, chapter_key, dst_name, _ in COPY_MAP:
        dst = os.path.join(CHAPTER_DIRS[chapter_key], dst_name)
        if _copy_if_exists(src, dst):
            print(f'  ✓  {dst_name}')
            copied += 1
        else:
            print(f'  ✗  {dst_name}  (source not found: {src})')
            missing += 1

    # Also try to copy newly regenerated 300-dpi versions
    extras = [
        ('./results/figures/trigger_pipeline_300.png',    'ch3_method',      'fig2_trigger_pipeline.png'),
        ('./results/figures/frequency_sharding_300.png',  'ch3_method',      'fig3_frequency_sharding.png'),
        ('./results/figures/frequency_comparison_300.png','ch3_method',      'fig4_frequency_comparison.png'),
        ('./results/figures/defense_evasion_concept_300.png', 'ch2_intro',   'fig1_defense_evasion_concept.png'),
        ('./results/figures/routing_mechanism.png',       'ch3_method',      'fig_routing_mechanism.png'),
        ('./results/figures/routing_comparison.png',      'ch3_method',      'fig_routing_comparison.png'),
        ('./results/figures/routing_scatter.png',         'ch5_appendix',    'figA4_routing_scatter.png'),
    ]
    for src, chapter_key, dst_name in extras:
        dst = os.path.join(CHAPTER_DIRS[chapter_key], dst_name)
        if _copy_if_exists(src, dst):
            print(f'  ✓  {dst_name}  (300 dpi regenerated)')
            copied += 1

    # Copy sensitivity / CIFAR-100 figures if they already exist
    for chapter_key, src, dst_name, _ in GENERATED_FIGURE_INDEX:
        if _copy_generated_if_exists(src, chapter_key, dst_name):
            print(f'  ✓  {dst_name}  (from {src})')
            copied += 1

    print(f'\n[P4-1] Step 2 complete: {copied} copied, {missing} missing.\n')
    return copied, missing


# ---------------------------------------------------------------------------
# Step 3 — Generate FIGURE_INDEX.md
# ---------------------------------------------------------------------------

FIGURE_INDEX_TEMPLATE = """\
# Thesis Figure Index  [P4-1]

> Auto-generated by `analysis/collect_figures.py`
> Last updated: {timestamp}
>
> All figures in `results/paper/` are 300 dpi, ready for thesis inclusion.
> Figures marked ⏳ require GPU training to be generated (run P2-5 first).

---

## Chapter 2 — Introduction (Motivation)

| No. | Filename | Description | Status |
|-----|----------|-------------|--------|
{ch2_rows}

---

## Chapter 3 — Method

| No. | Filename | Description | Status |
|-----|----------|-------------|--------|
{ch3_rows}

---

## Chapter 4 — Experiments

| No. | Filename | Description | Status |
|-----|----------|-------------|--------|
{ch4_rows}

---

## Chapter 5 — Appendix

| No. | Filename | Description | Status |
|-----|----------|-------------|--------|
{ch5_rows}

---

## Generation Commands

### No-GPU figures (regenerate anytime)
```bash
# All no-GPU figures in one shot
python analysis/collect_figures.py

# Individual scripts
python analysis/visualize_dual_routing.py          # routing_mechanism / comparison / scatter
python analysis/create_visualizations.py           # trigger pipeline, sharding, frequency
```

### GPU-required figures (run after P2-5)
```bash
# Main experiment (CIFAR-10, all 3 conditions)
python main.py --freq-strategy ANB  --defense-enabled 0          # ANB no defense
python main.py --freq-strategy FIXED --defense-enabled 1         # FIXED + FreqFed
python main.py --freq-strategy ANB  --defense-enabled 1          # ANB + FreqFed (ours)

# Sensitivity sweeps
python analysis/sensitivity.py --all

# CIFAR-100 generalization
python analysis/cifar100_experiment.py

# Re-plot sensitivity after training (no GPU needed)
python analysis/sensitivity.py --plot-only
python analysis/cifar100_experiment.py --plot-only
```

### Post-training collection (after GPU runs)
```bash
python analysis/collect_figures.py   # re-run to pick up new figures
```
"""


def _row(dst_name, description, chapter_dir, idx):
    path = os.path.join(chapter_dir, dst_name)
    status = '✅' if os.path.exists(path) else '⏳'
    return f'| {idx} | `{dst_name}` | {description} | {status} |'


def write_figure_index():
    """Write FIGURE_INDEX.md summarising all thesis figures."""
    print('[P4-1] Step 3: Writing FIGURE_INDEX.md...')

    # Build per-chapter rows
    chapter_entries = {k: [] for k in CHAPTER_DIRS}

    # From COPY_MAP
    for src, chapter_key, dst_name, desc in COPY_MAP:
        chapter_entries[chapter_key].append((dst_name, desc))

    # Extra 300-dpi regenerated versions (may override)
    regen_entries = {
        'ch3_method': [
            ('fig2_trigger_pipeline.png',   'ANB trigger generation pipeline (4-stage, 300 dpi)'),
            ('fig3_frequency_sharding.png', 'Frequency Sharding: per-client band allocation (300 dpi)'),
            ('fig4_frequency_comparison.png', 'Frequency domain comparison FIXED vs ANB (300 dpi)'),
            ('fig_routing_mechanism.png',   'Dual-Domain Routing pipeline for one image (300 dpi)'),
            ('fig_routing_comparison.png',  'Dual-Domain Routing across 4 texture levels (300 dpi)'),
        ],
        'ch2_intro': [
            ('fig1_defense_evasion_concept.png', 'Defense evasion motivation figure (300 dpi)'),
        ],
        'ch5_appendix': [
            ('figA4_routing_scatter.png', 'Dual-Domain Routing scatter: per-pixel weights (300 dpi)'),
        ],
    }
    for chapter_key, items in regen_entries.items():
        for dst_name, desc in items:
            # Update existing entry if present, else append
            existing = [e[0] for e in chapter_entries[chapter_key]]
            if dst_name not in existing:
                chapter_entries[chapter_key].append((dst_name, desc))

    # From GENERATED_FIGURE_INDEX
    for chapter_key, src, dst_name, desc in GENERATED_FIGURE_INDEX:
        existing = [e[0] for e in chapter_entries[chapter_key]]
        if dst_name not in existing:
            chapter_entries[chapter_key].append((dst_name, desc))

    def build_rows(chapter_key):
        entries = chapter_entries[chapter_key]
        if not entries:
            return '| — | — | (none) | — |'
        rows = []
        for i, (dst_name, desc) in enumerate(entries, 1):
            rows.append(_row(dst_name, desc, CHAPTER_DIRS[chapter_key], i))
        # Deduplicate by filename (keep last)
        seen, unique = set(), []
        for r in reversed(rows):
            fname = r.split('|')[2].strip()
            if fname not in seen:
                seen.add(fname)
                unique.append(r)
        return '\n'.join(reversed(unique))

    content = FIGURE_INDEX_TEMPLATE.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        ch2_rows=build_rows('ch2_intro'),
        ch3_rows=build_rows('ch3_method'),
        ch4_rows=build_rows('ch4_experiments'),
        ch5_rows=build_rows('ch5_appendix'),
    )

    idx_path = os.path.join(PAPER_DIR, 'FIGURE_INDEX.md')
    with open(idx_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'[P4-1] FIGURE_INDEX.md written → {idx_path}\n')
    return idx_path


# ---------------------------------------------------------------------------
# Step 4 — Print final inventory
# ---------------------------------------------------------------------------

def print_inventory():
    """Print a concise inventory of what's in results/paper/."""
    print('[P4-1] Final inventory of results/paper/:\n')
    total = 0
    for chapter_key, chapter_dir in sorted(CHAPTER_DIRS.items()):
        pngs = sorted(
            f for f in os.listdir(chapter_dir)
            if f.endswith('.png')
        ) if os.path.isdir(chapter_dir) else []
        print(f'  {chapter_dir}/')
        for p in pngs:
            size_kb = os.path.getsize(os.path.join(chapter_dir, p)) // 1024
            print(f'    {p}  ({size_kb} KB)')
            total += 1
        if not pngs:
            print('    (no figures yet)')
    print(f'\n  Total: {total} figure(s)\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='P4-1: Collect and organise thesis figures'
    )
    parser.add_argument(
        '--index-only', action='store_true',
        help='Only refresh FIGURE_INDEX.md, skip copying and regeneration'
    )
    parser.add_argument(
        '--skip-regen', action='store_true',
        help='Skip no-GPU figure regeneration (only copy existing files)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print('=' * 65)
    print('[P4-1] Thesis Figure Collection & Organisation')
    print('=' * 65)

    if args.index_only:
        write_figure_index()
        print_inventory()
        return

    if not args.skip_regen:
        regenerate_no_gpu_figures()

    collect_existing_figures()
    write_figure_index()
    print_inventory()

    print('=' * 65)
    print('[P4-1] Done. results/paper/ is ready for thesis inclusion.')
    print('       Run with --index-only after GPU experiments to refresh index.')
    print('=' * 65)


if __name__ == '__main__':
    main()
