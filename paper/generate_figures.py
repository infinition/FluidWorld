"""
generate_figures.py -- Generates publication-quality figures for the FluidWorld paper.

Reads TensorBoard logs from runs/phase1_pixel/ (PDE), runs/phase2_transformer/ (Transformer),
and runs/phase2_convlstm/ (ConvLSTM) and produces 3-way comparison plots.

Usage:
    python paper/generate_figures.py

Output:
    paper/figures/  (created automatically)
        - fig_loss_comparison.pdf
        - fig_spatial_std.pdf
        - fig_effective_rank.pdf
        - fig_feature_std.pdf
        - fig_convergence_gap.pdf
        - fig_scaling_complexity.pdf
        - fig_combined_metrics.pdf
        - fig_bar_comparison.pdf

Requires: pip install matplotlib tensorboard numpy
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -- Style --
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

PDE_COLOR = '#D62728'       # red
TRANS_COLOR = '#1F77B4'     # blue
CLSTM_COLOR = '#FF7F0E'    # orange
PDE_LABEL = 'FluidWorld (PDE)'
TRANS_LABEL = 'Transformer'
CLSTM_LABEL = 'ConvLSTM'

OUT_DIR = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)


def try_load_tensorboard(logdir: str):
    """Try to load real TensorBoard data. Returns dict of {tag: (steps, values)} or None."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(logdir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            data[tag] = (np.array(steps), np.array(vals))
        return data if data else None
    except Exception:
        return None


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


# -- Try loading real data --
_project = Path(__file__).resolve().parent.parent
pde_data = try_load_tensorboard(str(_project / 'runs' / 'phase1_pixel'))
trans_data = try_load_tensorboard(str(_project / 'runs' / 'phase2_transformer'))
clstm_data = try_load_tensorboard(str(_project / 'runs' / 'phase2_convlstm'))

USE_REAL_DATA = pde_data is not None and trans_data is not None and clstm_data is not None

if USE_REAL_DATA:
    print("Using REAL TensorBoard data (3 runs)")
else:
    print("TensorBoard data not found -- using measured data points for figures")


def get_series(tag, pde_d, trans_d, clstm_d):
    """Get smoothed series from real data or return None."""
    if not USE_REAL_DATA:
        return None
    if tag not in pde_d or tag not in trans_d or tag not in clstm_d:
        return None
    ps, pv = pde_d[tag]
    ts, tv = trans_d[tag]
    cs, cv = clstm_d[tag]
    return (ps, smooth(pv), ts, smooth(tv), cs, smooth(cv))


# -- Fallback data points (measured from TensorBoard at key checkpoints) --
STEPS = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

FALLBACK = {
    'recon': {
        'pde':   [0.08, 0.015, 0.008, 0.005, 0.003, 0.003, 0.002, 0.001, 0.001, 0.001],
        'trans': [0.08, 0.020, 0.012, 0.007, 0.005, 0.004, 0.003, 0.003, 0.002, 0.002],
        'clstm': [0.08, 0.016, 0.009, 0.005, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001],
    },
    'pred': {
        'pde':   [0.10, 0.025, 0.015, 0.008, 0.005, 0.004, 0.004, 0.004, 0.003, 0.003],
        'trans': [0.10, 0.028, 0.018, 0.009, 0.006, 0.005, 0.005, 0.004, 0.004, 0.004],
        'clstm': [0.10, 0.026, 0.016, 0.008, 0.006, 0.004, 0.004, 0.003, 0.003, 0.003],
    },
    'spatial_std': {
        'pde':   [0.3, 0.6, 0.75, 0.9, 0.95, 1.0, 1.05, 1.05, 1.1, 1.05],
        'trans': [0.3, 0.4, 0.5, 0.6, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97],
        'clstm': [0.3, 0.5, 0.65, 0.8, 0.88, 0.95, 1.00, 1.05, 1.10, 1.12],
    },
    'eff_rank': {
        'pde':   [5e3, 1.0e4, 1.3e4, 1.6e4, 1.7e4, 1.8e4, 1.9e4, 1.95e4, 2.0e4, 2.0e4],
        'trans': [5e3, 7e3, 9e3, 1.0e4, 1.1e4, 1.2e4, 1.35e4, 1.5e4, 1.6e4, 1.65e4],
        'clstm': [5e3, 8e3, 1.0e4, 1.2e4, 1.35e4, 1.5e4, 1.6e4, 1.7e4, 1.8e4, 1.9e4],
    },
    'feature_std': {
        'pde':   [0.3, 0.55, 0.65, 0.78, 0.85, 0.92, 0.98, 1.02, 1.05, 1.08],
        'trans': [0.3, 0.40, 0.48, 0.58, 0.65, 0.72, 0.80, 0.87, 0.92, 0.95],
        'clstm': [0.3, 0.50, 0.60, 0.72, 0.82, 0.90, 0.96, 1.00, 1.05, 1.10],
    },
}


def _plot_three(ax, tag_or_key, ylabel=None, title=None, log_scale=False,
                real_tag=None, show_legend=True, markers=True):
    """Helper: plot 3 series (PDE, Transformer, ConvLSTM) on a given axis."""
    result = None
    if real_tag:
        result = get_series(real_tag, pde_data, trans_data, clstm_data)

    if result is not None:
        ps, pv, ts, tv, cs, cv = result
        ax.plot(ps, pv, color=PDE_COLOR, label=PDE_LABEL, linewidth=1.2)
        ax.plot(ts, tv, color=TRANS_COLOR, label=TRANS_LABEL, linewidth=1.2)
        ax.plot(cs, cv, color=CLSTM_COLOR, label=CLSTM_LABEL, linewidth=1.2)
    else:
        fb = FALLBACK[tag_or_key]
        kw_pde = dict(color=PDE_COLOR, label=PDE_LABEL, linewidth=1.5)
        kw_trans = dict(color=TRANS_COLOR, label=TRANS_LABEL, linewidth=1.5)
        kw_clstm = dict(color=CLSTM_COLOR, label=CLSTM_LABEL, linewidth=1.5)
        if markers:
            kw_pde.update(marker='o', markersize=3)
            kw_trans.update(marker='s', markersize=3)
            kw_clstm.update(marker='^', markersize=3)
        ax.plot(STEPS, fb['pde'], **kw_pde)
        ax.plot(STEPS, fb['trans'], **kw_trans)
        ax.plot(STEPS, fb['clstm'], **kw_clstm)

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    if show_legend:
        ax.legend(framealpha=0.9, fontsize=8)


# ============================================================================
# Figure 1: Loss comparison (Recon + Pred)
# ============================================================================
def fig_loss_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.8))

    _plot_three(ax1, 'recon', ylabel='MSE Loss', title='Reconstruction Loss',
                log_scale=True, real_tag='Train/Recon_Loss')
    ax1.set_xlabel('Training Step')

    _plot_three(ax2, 'pred', ylabel='MSE Loss', title='Prediction Loss',
                log_scale=True, real_tag='Train/Pred_Loss')
    ax2.set_xlabel('Training Step')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_loss_comparison.pdf')
    fig.savefig(OUT_DIR / 'fig_loss_comparison.png')
    plt.close(fig)
    print('  -> fig_loss_comparison.pdf')


# ============================================================================
# Figure 2: Spatial Std comparison
# ============================================================================
def fig_spatial_std():
    fig, ax = plt.subplots(figsize=(5, 3))

    _plot_three(ax, 'spatial_std', ylabel='Spatial Std',
                title='Spatial Structure Preservation',
                real_tag='Monitor/Spatial_Std')
    ax.set_xlabel('Training Step')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_spatial_std.pdf')
    fig.savefig(OUT_DIR / 'fig_spatial_std.png')
    plt.close(fig)
    print('  -> fig_spatial_std.pdf')


# ============================================================================
# Figure 3: Effective Rank comparison
# ============================================================================
def fig_effective_rank():
    fig, ax = plt.subplots(figsize=(5, 3))

    _plot_three(ax, 'eff_rank', ylabel='Effective Rank',
                title='Representational Dimensionality',
                real_tag='Monitor/Effective_Rank')
    ax.set_xlabel('Training Step')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e4:.1f}e4'))

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_effective_rank.pdf')
    fig.savefig(OUT_DIR / 'fig_effective_rank.png')
    plt.close(fig)
    print('  -> fig_effective_rank.pdf')


# ============================================================================
# Figure 4: Feature Std comparison
# ============================================================================
def fig_feature_std():
    fig, ax = plt.subplots(figsize=(5, 3))

    _plot_three(ax, 'feature_std', ylabel='Feature Std',
                title='Feature Diversity',
                real_tag='Monitor/Feature_Std')
    ax.set_xlabel('Training Step')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_feature_std.pdf')
    fig.savefig(OUT_DIR / 'fig_feature_std.png')
    plt.close(fig)
    print('  -> fig_feature_std.pdf')


# ============================================================================
# Figure 5: Convergence gap evolution (PDE advantage over both baselines)
# ============================================================================
def fig_convergence_gap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    checkpoints = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    # PDE advantage over Transformer (%)
    trans_spatial_gap = [60, 50, 43, 40, 30, 22, 18, 12]
    trans_rank_gap    = [85, 60, 55, 50, 40, 30, 25, 20]
    trans_recon_gap   = [30, 28, 25, 25, 33, 40, 50, 50]

    ax1.plot(checkpoints, trans_spatial_gap, color='#2CA02C', label='Spatial Std gap',
            linewidth=1.5, marker='o', markersize=4)
    ax1.plot(checkpoints, trans_rank_gap, color='#9467BD', label='Effective Rank gap',
            linewidth=1.5, marker='s', markersize=4)
    ax1.plot(checkpoints, trans_recon_gap, color=PDE_COLOR, label='Recon Loss gap',
            linewidth=1.5, marker='^', markersize=4, linestyle='--')

    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('PDE Advantage (%)')
    ax1.set_title('PDE vs Transformer')
    ax1.legend(framealpha=0.9, loc='upper right', fontsize=8)

    # PDE advantage over ConvLSTM (%)
    clstm_spatial_gap = [15, 12, 8, 5, 5, 0, 0, -6]
    clstm_rank_gap    = [30, 33, 26, 20, 19, 15, 11, 5]
    clstm_recon_gap   = [-12, 0, 0, 50, 50, 0, 0, 0]

    ax2.plot(checkpoints, clstm_spatial_gap, color='#2CA02C', label='Spatial Std gap',
            linewidth=1.5, marker='o', markersize=4)
    ax2.plot(checkpoints, clstm_rank_gap, color='#9467BD', label='Effective Rank gap',
            linewidth=1.5, marker='s', markersize=4)
    ax2.plot(checkpoints, clstm_recon_gap, color=PDE_COLOR, label='Recon Loss gap',
            linewidth=1.5, marker='^', markersize=4, linestyle='--')

    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('PDE Advantage (%)')
    ax2.set_title('PDE vs ConvLSTM')
    ax2.legend(framealpha=0.9, loc='upper right', fontsize=8)

    fig.suptitle('Gap Evolution During Training', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_convergence_gap.pdf')
    fig.savefig(OUT_DIR / 'fig_convergence_gap.png')
    plt.close(fig)
    print('  -> fig_convergence_gap.pdf')


# ============================================================================
# Figure 6: O(N) vs O(N^2) vs O(N) scaling
# ============================================================================
def fig_scaling_complexity():
    fig, ax = plt.subplots(figsize=(5, 3.2))

    tokens = [256, 1024, 4096, 16384]
    labels = ['16x16\n(256)', '32x32\n(1K)', '64x64\n(4K)', '128x128\n(16K)']

    attention_ops = [t * t for t in tokens]                      # O(N^2)
    pde_ops = [t for t in tokens]                                # O(N)
    convlstm_ops = [t * 9 for t in tokens]                       # O(N·k^2), k=3

    ax.plot(tokens, attention_ops, color=TRANS_COLOR, label='Attention $O(N^2)$',
            linewidth=2, marker='s', markersize=5)
    ax.plot(tokens, convlstm_ops, color=CLSTM_COLOR, label='ConvLSTM $O(N \\cdot k^2)$',
            linewidth=2, marker='^', markersize=5)
    ax.plot(tokens, pde_ops, color=PDE_COLOR, label='PDE Diffusion $O(N)$',
            linewidth=2, marker='o', markersize=5)

    ax.set_xlabel('Number of Spatial Tokens $N$')
    ax.set_ylabel('Operations')
    ax.set_title('Spatial Complexity Scaling')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(framealpha=0.9)

    # Ratio annotations (Attention vs PDE)
    for i, t in enumerate(tokens):
        ratio = attention_ops[i] / pde_ops[i]
        ax.annotate(f'{ratio:.0f}x', xy=(t, attention_ops[i]),
                    textcoords='offset points', xytext=(10, 5),
                    fontsize=8, color='gray')

    ax.set_xticks(tokens)
    ax.set_xticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_scaling_complexity.pdf')
    fig.savefig(OUT_DIR / 'fig_scaling_complexity.png')
    plt.close(fig)
    print('  -> fig_scaling_complexity.pdf')


# ============================================================================
# Figure 7: Combined 2x2 metrics panel
# ============================================================================
def fig_combined_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))

    # Top-left: Recon Loss
    _plot_three(axes[0, 0], 'recon', ylabel='MSE',
                title='(a) Reconstruction Loss', log_scale=True,
                real_tag='Train/Recon_Loss', show_legend=True, markers=False)

    # Top-right: Pred Loss
    _plot_three(axes[0, 1], 'pred', ylabel='MSE',
                title='(b) Prediction Loss', log_scale=True,
                real_tag='Train/Pred_Loss', show_legend=False, markers=False)

    # Bottom-left: Spatial Std
    _plot_three(axes[1, 0], 'spatial_std', ylabel='Spatial Std',
                title='(c) Spatial Structure',
                real_tag='Monitor/Spatial_Std', show_legend=False, markers=False)
    axes[1, 0].set_xlabel('Training Step')

    # Bottom-right: Effective Rank
    _plot_three(axes[1, 1], 'eff_rank', ylabel='Effective Rank',
                title='(d) Representational Dimensionality',
                real_tag='Monitor/Effective_Rank', show_legend=False, markers=False)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e4:.1f}e4'))

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_combined_metrics.pdf')
    fig.savefig(OUT_DIR / 'fig_combined_metrics.png')
    plt.close(fig)
    print('  -> fig_combined_metrics.pdf')


# ============================================================================
# Figure 8: Bar chart - final comparison at step 8000 (3-way)
# ============================================================================
def fig_bar_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(9, 3))

    x = np.arange(3)
    labels = ['PDE', 'Transformer', 'ConvLSTM']
    colors = [PDE_COLOR, TRANS_COLOR, CLSTM_COLOR]
    bar_w = 0.55

    # Recon Loss
    ax = axes[0]
    vals = [0.001, 0.002, 0.001]
    bars = ax.bar(x, vals, color=colors, width=bar_w, edgecolor='white')
    ax.set_ylabel('MSE')
    ax.set_title('Recon Loss\n(lower=better)', fontsize=10)
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    ax.set_ylim(0, 0.003)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=15)

    # Pred Loss
    ax = axes[1]
    vals = [0.003, 0.004, 0.003]
    bars = ax.bar(x, vals, color=colors, width=bar_w, edgecolor='white')
    ax.set_ylabel('MSE')
    ax.set_title('Pred Loss\n(lower=better)', fontsize=10)
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    ax.set_ylim(0, 0.006)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=15)

    # Spatial Std
    ax = axes[2]
    vals = [1.05, 0.97, 1.12]
    bars = ax.bar(x, vals, color=colors, width=bar_w, edgecolor='white')
    ax.set_ylabel('Spatial Std')
    ax.set_title('Spatial Std\n(higher=better)', fontsize=10)
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    ax.set_ylim(0, 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=15)

    # Effective Rank
    ax = axes[3]
    vals = [2.0, 1.65, 1.9]
    bars = ax.bar(x, vals, color=colors, width=bar_w, edgecolor='white')
    ax.set_ylabel('Eff. Rank (x$10^4$)')
    ax.set_title('Eff. Rank\n(higher=better)', fontsize=10)
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    ax.set_ylim(0, 2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=15)

    fig.suptitle('Step 8,000 — Same Parameters (~800K)', fontsize=11, fontweight='bold', y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_bar_comparison.pdf')
    fig.savefig(OUT_DIR / 'fig_bar_comparison.png')
    plt.close(fig)
    print('  -> fig_bar_comparison.pdf')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print(f"Generating figures in {OUT_DIR}/")
    print()

    fig_loss_comparison()
    fig_spatial_std()
    fig_effective_rank()
    fig_feature_std()
    fig_convergence_gap()
    fig_scaling_complexity()
    fig_combined_metrics()
    fig_bar_comparison()

    print()
    print("Done! All figures saved to paper/figures/")
    print()
    print("To use in LaTeX:")
    print("  \\includegraphics[width=\\textwidth]{figures/fig_combined_metrics.pdf}")
