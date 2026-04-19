import matplotlib.pyplot as plt
import numpy as np

# ── Accurate Data from local_1.csv ────────────────────────────────────────────
sparsification = ['Baseline', 'K=5', 'K=10', 'K=25', 'K=50', 'K=100', 'K=250', 'K=500']
x = np.arange(len(sparsification))

datasets = {
    'Reddit': {
        'acc':  [0.9408, 0.7098, 0.7063, 0.7496, 0.8029, 0.8604, 0.9119, 0.9302],
        'time': [2342.29, 92.26, 97.81, 115.30, 179.04, 308.17, 703.10, 1204.02],
        'edges':[57424428, 689653, 1244952, 2835497, 5291891, 9649972, 19691977, 30598452],
    },
    'OGBN-Products': {
        'acc':  [0.9134, 0.7424, 0.7795, 0.8511, 0.8949, 0.9074, 0.9134, None],
        'time': [3015.11, 536.25, 549.51, 640.91, 877.31, 1758.35, 2678.49, None],
        'edges':[63083526, 6779472, 11450608, 22327577, 34705249, 47342947, 57076419, None],
    },
    'OGBN-Arxiv': {
        'acc':  [0.7064, 0.6109, 0.6481, 0.6800, 0.7000, 0.7038, 0.7066, None],
        'time': [49.69, 44.90, 43.33, 46.65, 46.94, 47.39, 48.72, None],
        'edges':[1242470, 417553, 608212, 871507, 1001256, 1078980, 1144817, None],
    },
    'Yelp': {
        'acc':  [0.8592, 0.8742, 0.8716, 0.8592, 0.8594, 0.8583, 0.8603, None],
        'time': [26.25, 12.66, 12.89, 12.87, 14.05, 18.54, 25.44, None],
        'edges':[3869956, 137598, 251340, 585767, 1112996, 2026430, 3626997, None],
    },
}

colors  = ['#E24B4A', '#378ADD', '#EF9F27', '#1D9E75']
markers = ['s', '^', 'o', 'D']

def split_none(xvals, yvals):
    """Split into segments separated by None, return list of (xs, ys) segments."""
    segments = []
    seg_x, seg_y = [], []
    for xi, yi in zip(xvals, yvals):
        if yi is not None:
            seg_x.append(xi)
            seg_y.append(yi)
        else:
            if seg_x:
                segments.append((seg_x, seg_y))
            seg_x, seg_y = [], []
    if seg_x:
        segments.append((seg_x, seg_y))
    return segments

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    'Edge Sparsification Results — Eigenvector Centrality\n'
    'Accuracy & Inference Time across Sparsification Factors',
    fontsize=13, fontweight='bold', y=1.03
)

# ── Left: Accuracy ────────────────────────────────────────────────────────────
ax1 = axes[0]
for i, (name, vals) in enumerate(datasets.items()):
    segs = split_none(x, vals['acc'])
    for si, (sx, sy) in enumerate(segs):
        ax1.plot(sx, sy,
                 color=colors[i], marker=markers[i],
                 markersize=7, linewidth=1.8, linestyle='--',
                 label=name if si == 0 else '_nolegend_')

ax1.set_title('Classification Accuracy', fontsize=12, pad=8)
ax1.set_xlabel('Sparsification Factor $k$', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(sparsification, rotation=25, ha='right', fontsize=9)
ax1.set_ylim(0.58, 0.98)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax1.legend(fontsize=9, framealpha=0.7, loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.spines[['top', 'right']].set_visible(False)

# ── Right: Time ───────────────────────────────────────────────────────────────
ax2 = axes[1]
for i, (name, vals) in enumerate(datasets.items()):
    segs = split_none(x, vals['time'])
    for si, (sx, sy) in enumerate(segs):
        ax2.plot(sx, sy,
                 color=colors[i], marker=markers[i],
                 markersize=7, linewidth=1.8, linestyle='--',
                 label=name if si == 0 else '_nolegend_')

ax2.set_title('Inference Time (seconds)', fontsize=12, pad=8)
ax2.set_xlabel('Sparsification Factor $k$', fontsize=11)
ax2.set_ylabel('Time (seconds)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(sparsification, rotation=25, ha='right', fontsize=9)
ax2.legend(fontsize=9, framealpha=0.7, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.45)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('local1_sparsification_plot.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved: local1_sparsification_plot.png")
