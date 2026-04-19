import matplotlib.pyplot as plt
import numpy as np

# ── Local Sparsification Data (local_1.csv) ───────────────────────────────────
# SF: Baseline, K=5, K=10, K=25, K=50, K=100, K=250, K=500
local_datasets = {
    'Reddit': {
        'acc':   [0.9408, 0.7098, 0.7063, 0.7496, 0.8029, 0.8604, 0.9119, 0.9302],
        'edges': [57424428, 689653, 1244952, 2835497, 5291891, 9649972, 19691977, 30598452],
    },
    'OGBN-Products': {
        'acc':   [0.9134, 0.7424, 0.7795, 0.8511, 0.8949, 0.9074, 0.9134, None],
        'edges': [63083526, 6779472, 11450608, 22327577, 34705249, 47342947, 57076419, None],
    },
    'OGBN-Arxiv': {
        'acc':   [0.7064, 0.6109, 0.6481, 0.6800, 0.7000, 0.7038, 0.7066, None],
        'edges': [1242470, 417553, 608212, 871507, 1001256, 1078980, 1144817, None],
    },
    'Yelp': {
        'acc':   [0.8592, 0.8742, 0.8716, 0.8592, 0.8594, 0.8583, 0.8603, None],
        'edges': [3869956, 137598, 251340, 585767, 1112996, 2026430, 3626997, None],
    },
}

# ── Global Sparsification Data (global_spaction.csv) ──────────────────────────
# SF: Baseline, 0.5, 0.6, 0.7, 0.8
global_datasets = {
    'Reddit': {
        'acc':   [0.9410, 0.8110, 0.8461, None,   0.8999],
        'edges': [57424428, 28770455, 34501250, 40232044, 45962839],
    },
    'OGBN-Products': {
        'acc':   [0.9139, 0.8589, 0.8699, 0.8829, 0.8974],
        'edges': [63083526, 32154014, 38339912, 44525811, 50711714],
    },
    'OGBN-Arxiv': {
        'acc':   [0.7086, 0.6178, 0.6272, 0.6515, 0.6680],
        'edges': [1242470, 663571, 779350, 895130, 1010910],
    },
    'Yelp': {
        'acc':   [0.8590, 0.8585, 0.8583, 0.8588, 0.8585],
        'edges': [3869956, 1946466, 2331164, 2715862, 3100560],
    },
}

colors  = ['#E24B4A', '#378ADD', '#EF9F27', '#1D9E75']
markers = ['s', '^', 'o', 'D']

def filter_none(edges, acc):
    pairs = [(e, a) for e, a in zip(edges, acc) if e is not None and a is not None]
    if not pairs:
        return [], []
    e, a = zip(*pairs)
    # sort by edges ascending
    sorted_pairs = sorted(zip(e, a), key=lambda p: p[0])
    return zip(*sorted_pairs)

# ── Figure: 2 rows x 2 cols ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Number of Edges vs Classification Accuracy\nLocal (top) & Global (bottom) Sparsification',
             fontsize=14, fontweight='bold', y=1.02)

titles_local  = ['Local Sparsification — Edges vs Accuracy (Reddit & OGBN-Products)',
                 'Local Sparsification — Edges vs Accuracy (OGBN-Arxiv & Yelp)']
titles_global = ['Global Sparsification — Edges vs Accuracy (Reddit & OGBN-Products)',
                 'Global Sparsification — Edges vs Accuracy (OGBN-Arxiv & Yelp)']

def plot_group(ax, datasets, dataset_names, colors, markers, title):
    baseline_handles = []
    for idx, name in enumerate(dataset_names):
        vals = datasets[name]
        ex, ay = filter_none(vals['edges'], vals['acc'])
        ex, ay = list(ex), list(ay)
        if not ex:
            continue

        # baseline is the point with the most edges (sorted ascending → last)
        baseline_edge = ex[-1]
        baseline_acc  = ay[-1]

        ax.plot(ex, ay,
                color=colors[idx], marker=markers[idx],
                markersize=7, linewidth=1.8, linestyle='--', label=name)

        # annotate every point with its accuracy
        for xi, yi in zip(ex, ay):
            ax.annotate(f'{yi:.3f}', xy=(xi, yi),
                        textcoords='offset points', xytext=(4, 5),
                        fontsize=7, color=colors[idx])

        # ── Baseline marker ───────────────────────────────────────────────────
        bl = ax.plot(baseline_edge, baseline_acc,
                     marker='*', color=colors[idx],
                     markersize=16, zorder=6,
                     markeredgecolor='black', markeredgewidth=0.6,
                     linestyle='None',
                     label=f'{name} Baseline ({baseline_acc:.4f})')
        baseline_handles.append(bl[0])

        ax.annotate(f'Baseline\n{baseline_acc:.4f}',
                    xy=(baseline_edge, baseline_acc),
                    textcoords='offset points', xytext=(-52, -22),
                    fontsize=7.5, color=colors[idx], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=colors[idx], lw=0.8))

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel('Number of Edges', fontsize=10)
    ax.set_ylabel('Classification Accuracy', fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

    # combined legend: dataset lines + baseline stars
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=8, framealpha=0.7,
              ncol=1, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.spines[['top', 'right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

# Row 0: Local
plot_group(axes[0][0], local_datasets,
           ['Reddit', 'OGBN-Products'], colors[0:2], markers[0:2],
           'Local — Reddit & OGBN-Products')
plot_group(axes[0][1], local_datasets,
           ['OGBN-Arxiv', 'Yelp'], colors[2:4], markers[2:4],
           'Local — OGBN-Arxiv & Yelp')

# Row 1: Global
plot_group(axes[1][0], global_datasets,
           ['Reddit', 'OGBN-Products'], colors[0:2], markers[0:2],
           'Global — Reddit & OGBN-Products')
plot_group(axes[1][1], global_datasets,
           ['OGBN-Arxiv', 'Yelp'], colors[2:4], markers[2:4],
           'Global — OGBN-Arxiv & Yelp')

plt.tight_layout()
plt.savefig('edges_vs_accuracy.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved: edges_vs_accuracy.png")
