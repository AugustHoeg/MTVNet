import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Global font settings (Times/serif, larger size)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 26,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22
})
metric_fontsize = 18

# New model order
models = ["RCAN", "HAT", "ArSSR", "EDDSR",
          "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"]

# Corresponding DI values reordered
hcp_1200    = [8.9, 10.0, 7.2, 6.1, 7.2, 7.5, 12.9, 10.3, 12.9]
facts_synth = [10.7, 11.7, 7.7, 6.1, 11.5, 12.4, 13.7, 25.0, 27.3]

# Find best (max) and second-best indices, handling ties
def get_best_indices(values):
    arr = np.array(values)
    max_val = arr.max()
    best = np.where(arr == max_val)[0].tolist()
    mask = arr != max_val
    if mask.any():
        second_val = arr[mask].max()
        second = np.where(arr == second_val)[0].tolist()
    else:
        second = []
    return best, second

best_hcp, second_hcp = get_best_indices(hcp_1200)
best_facts, second_facts = get_best_indices(facts_synth)

# Function to underline text using Unicode combining underline
def underline_text(s):
    return ''.join([c + '\u0332' for c in s])

# Annotate bars with best/second-best highlighted
def annotate_bars(ax, bars, best_idx_list, second_idx_list):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i in best_idx_list:
            text = f"$\\mathbf{{{height:.1f}}}$"
        elif i in second_idx_list:
            text = underline_text(f"{height:.1f}")
        else:
            text = f"{height:.1f}"
        ax.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=metric_fontsize)

# Colors
colors = ["burlywood", "darkseagreen"]

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
x_pos = np.arange(len(models))
bar_width = 0.7

# --- Left subplot: HCP-1200 ---
bars1 = ax1.bar(x_pos, hcp_1200, width=bar_width, color=colors[0],
                edgecolor="black", linewidth=1, label="HCP-1200")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=35, ha="right")
ax1.set_ylabel("Mean Diffusion Index")
ax1.set_ylim(0, 30)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_axisbelow(True)
ax1.legend(loc="upper left", frameon=False)
annotate_bars(ax1, bars1, best_hcp, second_hcp)

# --- Right subplot: FACTS-Synth ---
bars2 = ax2.bar(x_pos, facts_synth, width=bar_width, color=colors[1],
                edgecolor="black", linewidth=1, label="FACTS-Synth")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, rotation=35, ha="right")
ax2.set_ylim(0, 30)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_axisbelow(True)
ax2.legend(loc="upper left", frameon=False)
annotate_bars(ax2, bars2, best_facts, second_facts)

# Remove top spine, keep left
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.25, wspace=0.025)
plt.savefig("DI_barplot_subfigures_legends_reordered.pdf", dpi=600, bbox_inches="tight")
plt.show()
