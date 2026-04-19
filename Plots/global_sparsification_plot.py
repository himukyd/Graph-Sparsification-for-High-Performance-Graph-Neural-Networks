import matplotlib.pyplot as plt
import numpy as np

# ── Accurate Data from global_spaction.csv ────────────────────────────────────
# Sparsification factors: Baseline, 0.5, 0.6, 0.7, 0.8
sparsification = ['Baseline', 'K=0.5', 'K=0.6', 'K=0.7', 'K=0.8']
x = np.arange(len(sparsification))

# Note: Reddit K=0.7 has only edges, Acc & Time are missing → None
datasets = {
    'Reddit': {
        'acc':  [0.9410, 0.8110, 0.8461, 0.8778,   0.8999],
        'time': [2656.42, 1562.73, 1906.34, 2187.54, 2532.62],
        'edges':[57424428, 28770455, 34501250, 40232044, 45962839],
    },
    'OGBN-Products': {
        'acc':  [0.9139, 0.8589, 0.8699, 0.8829, 0.8974],
        'time': [2987.80, 1904.79, 2143.35, 2376.72, 2602.76],
        'edges':[63083526, 32154014, 38339912, 44525811, 50711714],
    },
    'OGBN-Arxiv': {
        'acc':  [0.7086, 0.6178, 0.6272, 0.6515, 0.6680],
        'time': [60.47, 57.49, 58.64, 60.09, 59.17],
        'edges':[1242470, 663571, 779350, 895130, 1010910],
    },
    'Yelp': {
        'acc':  [0.8590, 0.8585, 0.8583, 0.8588, 0.8585],
        'time': [26.13, 18.17, 20.60, 21.777, 23.355],
        'edges':[3869956, 1946466, 2331164, 2715862, 3100560],
    },
}

colors  = ['#E24B4A', '#378ADD', '#EF9F27', '#1D9E75']
markers = ['s', '^', 'o', 'D']

def split_none(xvals, yvals):
    """Split into continuous segments around None values."""
    segments = []
    seg_x, seg_y = [], []
    for xi, yi in zip(xvals, yvals):
        if yi is not None:
            seg_x.append(xi)
            seg_y.append(yi)
        else:
            if seg_x:
                segments.append((seg_x[:], seg_y[:]))
            seg_x, seg_y = [], []
    if seg_x:
        segments.append((seg_x, seg_y))
    return segments

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    'Global Edge Sparsification Results\n'
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
    # Mark missing points with a hollow circle
    for xi, yi in zip(x, vals['acc']):
        if yi is None:
            ax1.plot(xi, ax1.get_ylim()[0],  # placeholder; actual mark below after ylim set
                     marker='x', color=colors[i], markersize=8, zorder=5)

ax1.set_title('Classification Accuracy', fontsize=12, pad=8)
ax1.set_xlabel('Sparsification Factor', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(sparsification, fontsize=10)
ax1.set_ylim(0.58, 0.98)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax1.legend(fontsize=9, framealpha=0.7, loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.spines[['top', 'right']].set_visible(False)

# Annotate missing point for Reddit K=0.7
ax1.annotate('N/A', xy=(3, 0.60), fontsize=8, color=colors[0],
             ha='center', style='italic')

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
ax2.set_xlabel('Sparsification Factor', fontsize=11)
ax2.set_ylabel('Time (seconds)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(sparsification, fontsize=10)
ax2.legend(fontsize=9, framealpha=0.7, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.45)
ax2.spines[['top', 'right']].set_visible(False)

# Annotate missing point for Reddit K=0.7
ax2.annotate('N/A', xy=(3, 100), fontsize=8, color=colors[0],
             ha='center', style='italic')

plt.tight_layout()
plt.savefig('global_sparsification_plot.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved: global_sparsification_plot.png")
