"""
FedRGBD — Publication-Quality Convergence Plots (v2)
IEEE Sensors Journal-ready figures with external legends, high readability.

Usage: python3 generate_plots.py
Output: figures/ directory with PDF and PNG plots
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# ALL EXPERIMENT RESULTS
# ============================================================

results = {
    "2N-IID-FedAvg": {
        "rounds": [1, 2, 3],
        "accuracy": [98.79, 99.60, 99.68],
        "loss": [0.0384, 0.0136, 0.0090],
        "time_s": 3673,
        "nodes": 2, "distribution": "IID", "strategy": "FedAvg",
    },
    "2N-NonIID-FedAvg": {
        "rounds": [1, 2, 3],
        "accuracy": [78.36, 99.60, 99.85],
        "loss": [0.6401, 0.0158, 0.0050],
        "time_s": 4867,
        "nodes": 2, "distribution": "Non-IID", "strategy": "FedAvg",
    },
    "3N-IID-FedAvg": {
        "rounds": [1, 2, 3],
        "accuracy": [85.45, 99.69, 99.75],
        "loss": [0.3364, 0.0214, 0.0113],
        "time_s": 4560,
        "nodes": 3, "distribution": "IID", "strategy": "FedAvg",
        "per_node_r1": {"A": 84.89, "B": 85.58, "C": 86.33},
        "per_node_r3": {"A": 99.69, "B": 99.78, "C": 99.79},
    },
    "3N-NonIID-FedAvg": {
        "rounds": [1, 2, 3],
        "accuracy": [53.66, 99.15, 99.10],
        "loss": [0.7849, 0.1156, 0.0420],
        "time_s": 6174,
        "nodes": 3, "distribution": "Non-IID", "strategy": "FedAvg",
        "per_node_r1": {"A": 45.11, "B": 48.69, "C": 85.92},
        "per_node_r3": {"A": 99.02, "B": 98.92, "C": 99.71},
    },
    "3N-NonIID-FedProx001": {
        "rounds": [1, 2, 3],
        "accuracy": [94.79, 99.27, 99.49],
        "loss": [0.1451, 0.0264, 0.0207],
        "time_s": 10080,
        "nodes": 3, "distribution": "Non-IID", "strategy": r"FedProx ($\mu$=0.01)",
        "per_node_r1": {"A": 93.48, "B": 94.61, "C": 98.33},
        "per_node_r3": {"A": 99.58, "B": 99.37, "C": 99.58},
    },
    "3N-NonIID-FedProx01": {
        "rounds": [1, 2, 3],
        "accuracy": [96.91, 97.81, 98.35],
        "loss": [0.0930, 0.0627, 0.0467],
        "time_s": 9298,
        "nodes": 3, "distribution": "Non-IID", "strategy": r"FedProx ($\mu$=0.1)",
        "per_node_r1": {"A": 96.93, "B": 97.27, "C": 96.00},
        "per_node_r3": {"A": 98.21, "B": 98.07, "C": 99.38},
    },
}

# ============================================================
# STYLING
# ============================================================

IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
}

COLORS = {
    "FedAvg-IID":           "#1565C0",  # Dark blue
    "FedAvg-NonIID":        "#C62828",  # Dark red
    "FedProx001-NonIID":    "#2E7D32",  # Dark green
    "FedProx01-NonIID":     "#E65100",  # Dark orange
    "FedAvg-IID-2N":        "#64B5F6",  # Light blue
    "FedAvg-NonIID-2N":     "#EF9A9A",  # Light red
}

MARKERS = {
    "FedAvg-IID":           "o",
    "FedAvg-NonIID":        "s",
    "FedProx001-NonIID":    "^",
    "FedProx01-NonIID":     "D",
    "FedAvg-IID-2N":        "o",
    "FedAvg-NonIID-2N":     "s",
}

LINESTYLES = {
    "FedAvg-IID":           "-",
    "FedAvg-NonIID":        "-",
    "FedProx001-NonIID":    "--",
    "FedProx01-NonIID":     "-.",
    "FedAvg-IID-2N":        "--",
    "FedAvg-NonIID-2N":     "--",
}


def setup_style():
    plt.rcParams.update(IEEE_STYLE)


def fig1_convergence_accuracy(save_dir):
    """Figure 1: Global accuracy convergence — all 3-node strategies."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    plot_configs = [
        ("3N-IID-FedAvg",          "FedAvg (IID)",                    "FedAvg-IID"),
        ("3N-NonIID-FedAvg",       "FedAvg (Non-IID)",                "FedAvg-NonIID"),
        ("3N-NonIID-FedProx001",   r"FedProx $\mu$=0.01 (Non-IID)",  "FedProx001-NonIID"),
        ("3N-NonIID-FedProx01",    r"FedProx $\mu$=0.1 (Non-IID)",   "FedProx01-NonIID"),
    ]

    for key, label, style_key in plot_configs:
        r = results[key]
        ax.plot(r["rounds"], r["accuracy"],
                color=COLORS[style_key],
                marker=MARKERS[style_key],
                linestyle=LINESTYLES[style_key],
                label=label,
                markerfacecolor='white',
                markeredgewidth=2.0,
                zorder=3)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim([45, 102])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc',
              columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(os.path.join(save_dir, "fig1_convergence_accuracy.pdf"))
    fig.savefig(os.path.join(save_dir, "fig1_convergence_accuracy.png"))
    plt.close()
    print("  [OK] fig1_convergence_accuracy")


def fig2_convergence_loss(save_dir):
    """Figure 2: Loss convergence — all 3-node strategies."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    plot_configs = [
        ("3N-IID-FedAvg",          "FedAvg (IID)",                    "FedAvg-IID"),
        ("3N-NonIID-FedAvg",       "FedAvg (Non-IID)",                "FedAvg-NonIID"),
        ("3N-NonIID-FedProx001",   r"FedProx $\mu$=0.01 (Non-IID)",  "FedProx001-NonIID"),
        ("3N-NonIID-FedProx01",    r"FedProx $\mu$=0.1 (Non-IID)",   "FedProx01-NonIID"),
    ]

    for key, label, style_key in plot_configs:
        r = results[key]
        ax.plot(r["rounds"], r["loss"],
                color=COLORS[style_key],
                marker=MARKERS[style_key],
                linestyle=LINESTYLES[style_key],
                label=label,
                markerfacecolor='white',
                markeredgewidth=2.0,
                zorder=3)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Distributed Loss")
    ax.set_xticks([1, 2, 3])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc',
              columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(os.path.join(save_dir, "fig2_convergence_loss.pdf"))
    fig.savefig(os.path.join(save_dir, "fig2_convergence_loss.png"))
    plt.close()
    print("  [OK] fig2_convergence_loss")


def fig3_2node_vs_3node(save_dir):
    """Figure 3: 2-Node vs 3-Node comparison (IID and Non-IID side by side)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Left: IID
    for key, label, style_key in [
        ("2N-IID-FedAvg", "2-Node", "FedAvg-IID-2N"),
        ("3N-IID-FedAvg", "3-Node", "FedAvg-IID"),
    ]:
        r = results[key]
        ax1.plot(r["rounds"], r["accuracy"],
                 color=COLORS[style_key],
                 marker=MARKERS[style_key],
                 linestyle=LINESTYLES[style_key],
                 label=label,
                 markerfacecolor='white',
                 markeredgewidth=2.0)

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Global Test Accuracy (%)")
    ax1.set_title("(a) IID Distribution", fontweight='bold', pad=8)
    ax1.set_xticks([1, 2, 3])
    ax1.set_ylim([80, 101])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc')

    # Right: Non-IID
    for key, label, style_key in [
        ("2N-NonIID-FedAvg", "2-Node", "FedAvg-NonIID-2N"),
        ("3N-NonIID-FedAvg", "3-Node", "FedAvg-NonIID"),
    ]:
        r = results[key]
        ax2.plot(r["rounds"], r["accuracy"],
                 color=COLORS[style_key],
                 marker=MARKERS[style_key],
                 linestyle=LINESTYLES[style_key],
                 label=label,
                 markerfacecolor='white',
                 markeredgewidth=2.0)

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Global Test Accuracy (%)")
    ax2.set_title("(b) Non-IID Label Skew", fontweight='bold', pad=8)
    ax2.set_xticks([1, 2, 3])
    ax2.set_ylim([45, 101])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(save_dir, "fig3_2node_vs_3node.pdf"))
    fig.savefig(os.path.join(save_dir, "fig3_2node_vs_3node.png"))
    plt.close()
    print("  [OK] fig3_2node_vs_3node")


def fig4_per_node_noniid(save_dir):
    """Figure 4: Per-node accuracy — FedAvg vs FedProx μ=0.01 vs μ=0.1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    nodes = ["Node A\n(80% Fire)", "Node B\n(88.5% Fire)", "Node C\n(20% Fire)"]
    node_keys = ["A", "B", "C"]
    x = np.arange(len(nodes))
    width = 0.25

    # --- Left: Round 1 ---
    fedavg_r1 = [results["3N-NonIID-FedAvg"]["per_node_r1"][k] for k in node_keys]
    fp001_r1 = [results["3N-NonIID-FedProx001"]["per_node_r1"][k] for k in node_keys]
    fp01_r1 = [results["3N-NonIID-FedProx01"]["per_node_r1"][k] for k in node_keys]

    b1 = ax1.bar(x - width, fedavg_r1, width, label="FedAvg",
                 color=COLORS["FedAvg-NonIID"], alpha=0.85, edgecolor='white', linewidth=0.5)
    b2 = ax1.bar(x, fp001_r1, width, label=r"FedProx $\mu$=0.01",
                 color=COLORS["FedProx001-NonIID"], alpha=0.85, edgecolor='white', linewidth=0.5)
    b3 = ax1.bar(x + width, fp01_r1, width, label=r"FedProx $\mu$=0.1",
                 color=COLORS["FedProx01-NonIID"], alpha=0.85, edgecolor='white', linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 1.2,
                     f'{h:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Round 1 — Initial Convergence", fontweight='bold', pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nodes, fontsize=9)
    ax1.set_ylim([0, 115])
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax1.set_axisbelow(True)

    # --- Right: Round 3 ---
    fedavg_r3 = [results["3N-NonIID-FedAvg"]["per_node_r3"][k] for k in node_keys]
    fp001_r3 = [results["3N-NonIID-FedProx001"]["per_node_r3"][k] for k in node_keys]
    fp01_r3 = [results["3N-NonIID-FedProx01"]["per_node_r3"][k] for k in node_keys]

    b4 = ax2.bar(x - width, fedavg_r3, width, label="FedAvg",
                 color=COLORS["FedAvg-NonIID"], alpha=0.85, edgecolor='white', linewidth=0.5)
    b5 = ax2.bar(x, fp001_r3, width, label=r"FedProx $\mu$=0.01",
                 color=COLORS["FedProx001-NonIID"], alpha=0.85, edgecolor='white', linewidth=0.5)
    b6 = ax2.bar(x + width, fp01_r3, width, label=r"FedProx $\mu$=0.1",
                 color=COLORS["FedProx01-NonIID"], alpha=0.85, edgecolor='white', linewidth=0.5)

    for bars in [b4, b5, b6]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b) Round 3 — Final Accuracy", fontweight='bold', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nodes, fontsize=9)
    ax2.set_ylim([97.5, 100.3])
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax2.set_axisbelow(True)

    # Shared legend below both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, frameon=True, framealpha=0.95, edgecolor='#cccccc',
               fontsize=9, handletextpad=0.5, columnspacing=1.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(os.path.join(save_dir, "fig4_per_node_noniid.pdf"))
    fig.savefig(os.path.join(save_dir, "fig4_per_node_noniid.png"))
    plt.close()
    print("  [OK] fig4_per_node_noniid")


def fig5_training_time(save_dir):
    """Figure 5: Training time and final accuracy comparison."""
    fig, ax1 = plt.subplots(figsize=(5.0, 3.2))

    order = [
        ("2N-IID-FedAvg",          "2N IID\nFedAvg",                     COLORS["FedAvg-IID-2N"]),
        ("2N-NonIID-FedAvg",       "2N Non-IID\nFedAvg",                 COLORS["FedAvg-NonIID-2N"]),
        ("3N-IID-FedAvg",          "3N IID\nFedAvg",                     COLORS["FedAvg-IID"]),
        ("3N-NonIID-FedAvg",       "3N Non-IID\nFedAvg",                 COLORS["FedAvg-NonIID"]),
        ("3N-NonIID-FedProx001",   "3N Non-IID\n" + r"FedProx $\mu$=0.01", COLORS["FedProx001-NonIID"]),
        ("3N-NonIID-FedProx01",    "3N Non-IID\n" + r"FedProx $\mu$=0.1",  COLORS["FedProx01-NonIID"]),
    ]

    labels = [o[1] for o in order]
    times = [results[o[0]]["time_s"] / 60 for o in order]
    colors = [o[2] for o in order]
    accs = [results[o[0]]["accuracy"][-1] for o in order]

    x = np.arange(len(labels))

    # Bar chart for time
    bars = ax1.bar(x, times, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel("Total Training Time (min)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax1.set_axisbelow(True)

    # Time labels on bars
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                 f'{t:.0f} min', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Secondary axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(x, accs, color='#333333', marker='*', markersize=12, linestyle='--',
             linewidth=1.5, label='Final Accuracy', zorder=5)
    ax2.set_ylabel("Final Accuracy (%)")
    ax2.set_ylim([97.5, 100.5])

    # Accuracy labels
    for xi, acc in zip(x, accs):
        ax2.annotate(f'{acc:.2f}%', (xi, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=7, fontweight='bold', color='#333333')

    # Legend for accuracy line
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
               frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=9)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(save_dir, "fig5_time_vs_accuracy.pdf"))
    fig.savefig(os.path.join(save_dir, "fig5_time_vs_accuracy.png"))
    plt.close()
    print("  [OK] fig5_time_vs_accuracy")


def fig6_mu_tradeoff(save_dir):
    """Figure 6: FedProx μ trade-off — R1 accuracy vs R3 accuracy vs time."""
    fig, ax1 = plt.subplots(figsize=(4.5, 3.2))

    mu_labels = [r"FedAvg ($\mu$=0)", r"FedProx ($\mu$=0.01)", r"FedProx ($\mu$=0.1)"]
    mu_vals = [0, 0.01, 0.1]
    r1_accs = [
        results["3N-NonIID-FedAvg"]["accuracy"][0],
        results["3N-NonIID-FedProx001"]["accuracy"][0],
        results["3N-NonIID-FedProx01"]["accuracy"][0],
    ]
    r3_accs = [
        results["3N-NonIID-FedAvg"]["accuracy"][2],
        results["3N-NonIID-FedProx001"]["accuracy"][2],
        results["3N-NonIID-FedProx01"]["accuracy"][2],
    ]

    x = np.arange(len(mu_labels))

    ax1.plot(x, r1_accs, color=COLORS["FedAvg-NonIID"], marker='s', markersize=10,
             linewidth=2.0, label='Round 1 Accuracy', markerfacecolor='white', markeredgewidth=2.0)
    ax1.plot(x, r3_accs, color=COLORS["FedProx001-NonIID"], marker='o', markersize=10,
             linewidth=2.0, label='Round 3 Accuracy', markerfacecolor='white', markeredgewidth=2.0)

    # Annotations
    for i, (r1, r3) in enumerate(zip(r1_accs, r3_accs)):
        ax1.annotate(f'{r1:.1f}%', (i, r1), textcoords="offset points",
                     xytext=(-10, 10), ha='center', fontsize=8, color=COLORS["FedAvg-NonIID"],
                     fontweight='bold')
        ax1.annotate(f'{r3:.2f}%', (i, r3), textcoords="offset points",
                     xytext=(10, -14), ha='center', fontsize=8, color=COLORS["FedProx001-NonIID"],
                     fontweight='bold')

    ax1.set_xlabel("FL Strategy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mu_labels, fontsize=9)
    ax1.set_ylim([45, 102])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Shade the optimal region
    ax1.axvspan(0.7, 1.3, alpha=0.08, color=COLORS["FedProx001-NonIID"])
    ax1.text(1, 50, r'Optimal $\mu$', ha='center', fontsize=9, fontstyle='italic',
             color=COLORS["FedProx001-NonIID"], alpha=0.7)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc',
               fontsize=9, handletextpad=0.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(save_dir, "fig6_mu_tradeoff.pdf"))
    fig.savefig(os.path.join(save_dir, "fig6_mu_tradeoff.png"))
    plt.close()
    print("  [OK] fig6_mu_tradeoff")


def print_summary_table():
    """Print full summary table to console."""
    print("\n" + "=" * 100)
    print("  FedRGBD EXPERIMENT RESULTS SUMMARY — ALL 6 CONFIGURATIONS")
    print("=" * 100)
    print(f"{'Experiment':<28} {'R1 Acc':>8} {'R2 Acc':>8} {'R3 Acc':>8} "
          f"{'R1 Loss':>8} {'R3 Loss':>8} {'Time':>8}")
    print("-" * 100)

    for name, r in results.items():
        t_min = r['time_s'] / 60
        print(f"{name:<28} {r['accuracy'][0]:>7.2f}% {r['accuracy'][1]:>7.2f}% "
              f"{r['accuracy'][2]:>7.2f}% {r['loss'][0]:>8.4f} {r['loss'][2]:>8.4f} {t_min:>6.0f}m")

    print("-" * 100)

    print("\nKEY FINDINGS:")
    print(f"  1. Non-IID R1 penalty (3N FedAvg):    "
          f"-{results['3N-IID-FedAvg']['accuracy'][0] - results['3N-NonIID-FedAvg']['accuracy'][0]:.1f}% "
          f"(85.5% -> 53.7%)")
    print(f"  2. FedProx mu=0.01 R1 recovery:       "
          f"+{results['3N-NonIID-FedProx001']['accuracy'][0] - results['3N-NonIID-FedAvg']['accuracy'][0]:.1f}% "
          f"(53.7% -> 94.8%)")
    print(f"  3. FedProx mu=0.1 R1 (even better):   "
          f"+{results['3N-NonIID-FedProx01']['accuracy'][0] - results['3N-NonIID-FedAvg']['accuracy'][0]:.1f}% "
          f"(53.7% -> 96.9%)")
    print(f"  4. OVER-REGULARIZATION at mu=0.1:     "
          f"R3 = {results['3N-NonIID-FedProx01']['accuracy'][2]:.2f}% vs "
          f"{results['3N-NonIID-FedProx001']['accuracy'][2]:.2f}% (mu=0.01)")
    print(f"  5. OPTIMAL: mu=0.01 gives BEST R3:    "
          f"{results['3N-NonIID-FedProx001']['accuracy'][2]:.2f}% "
          f"(highest final accuracy)")
    print(f"  6. Time overhead FedProx vs FedAvg:    "
          f"+{(results['3N-NonIID-FedProx001']['time_s'] - results['3N-NonIID-FedAvg']['time_s']) / 60:.0f} min "
          f"(mu=0.01), "
          f"+{(results['3N-NonIID-FedProx01']['time_s'] - results['3N-NonIID-FedAvg']['time_s']) / 60:.0f} min "
          f"(mu=0.1)")
    print(f"  7. 2N vs 3N scaling:                   "
          f"R1 IID gap = {results['2N-IID-FedAvg']['accuracy'][0] - results['3N-IID-FedAvg']['accuracy'][0]:.1f}%, "
          f"R1 Non-IID gap = {results['2N-NonIID-FedAvg']['accuracy'][0] - results['3N-NonIID-FedAvg']['accuracy'][0]:.1f}%")
    print()


def main():
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)

    setup_style()
    print("Generating FedRGBD publication figures (v2)...")
    print(f"  Output: {save_dir}/\n")

    fig1_convergence_accuracy(save_dir)
    fig2_convergence_loss(save_dir)
    fig3_2node_vs_3node(save_dir)
    fig4_per_node_noniid(save_dir)
    fig5_training_time(save_dir)
    fig6_mu_tradeoff(save_dir)

    print(f"\nAll 6 figures saved to {save_dir}/")
    print_summary_table()


if __name__ == "__main__":
    main()
