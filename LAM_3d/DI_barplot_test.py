import matplotlib.pyplot as plt
import numpy as np

# Global font settings (Times/serif, larger size)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,          # base font size
    "axes.titlesize": 20,     # title
    "axes.labelsize": 20,     # x/y labels
    "xtick.labelsize": 20,    # tick labels
    "ytick.labelsize": 20,
    "legend.fontsize": 20
})

# Data
models = [
    "HAT", "RCAN", "EDDSR", "ArSSR",
    "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"
]

#facts_synth = [11.66, 10.66, 6.12, 7.74, 11.49, 12.37, 13.69, 24.98, 27.30]
#hcp_1200    = [10.02, 8.91, 6.08, 7.24, 7.22, 7.50, 12.93, 10.33, 12.93]

facts_synth = [11.7, 10.7, 6.1, 7.7, 11.5, 12.4, 13.7, 25.0, 27.3]
hcp_1200    = [10.0, 8.9, 6.1, 7.2, 7.2, 7.5, 12.9, 10.3, 12.9]

# Find best (max) and second-best indices, handling ties
def get_best_indices(values):
    arr = np.array(values)
    max_val = arr.max()
    best = np.where(arr == max_val)[0].tolist()

    # Mask out best values to find the next max
    mask = arr != max_val
    if mask.any():
        second_val = arr[mask].max()
        second = np.where(arr == second_val)[0].tolist()
    else:
        second = []
    return best, second

best_hcp, second_hcp = get_best_indices(hcp_1200)
best_facts, second_facts = get_best_indices(facts_synth)

# Bar positions
x = np.arange(len(models))
width = 0.35
gap = 0.1  # separation between bar pairs

# Colors (HCP-1200 first, then FACTS-Synth)
colors = ["burlywood", "darkseagreen"]

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
bars1 = ax.bar(x - width/2 - gap/2, hcp_1200, width, label="HCP-1200",
               color=colors[0], edgecolor="black", linewidth=1)
bars2 = ax.bar(x + width/2 + gap/2, facts_synth, width, label="FACTS-Synth",
               color=colors[1], edgecolor="black", linewidth=1)

# Labels and styling
ax.set_ylabel("Mean Diffusion Index")
#ax.set_title("Mean Diffusion Index across Models and Datasets", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.set_ylim(0, 30)

# Add grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_axisbelow(True)

# Legend at bottom
ax.legend(loc="upper left", ncol=2, frameon=False)

# Function to underline text using Unicode combining underline
def underline_text(s):
    return ''.join([c + '\u0332' for c in s])

# Annotate bars with best/second-best highlighted
def annotate_bars(bars, best_idx_list, second_idx_list, ha="left"):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i in best_idx_list:
            text = f"$\\mathbf{{{height:.1f}}}$"   # bold
        elif i in second_idx_list:
            text = underline_text(f"{height:.1f}")  # underline
        else:
            text = f"{height:.1f}"
        ax.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    va="bottom", fontsize=18, rotation=0, ha=ha)

annotate_bars(bars1, best_hcp, second_hcp, ha="center")       # HCP-1200
annotate_bars(bars2, best_facts, second_facts, ha="center")   # FACTS-Synth

plt.tight_layout()
plt.savefig("DI_barplot_large_fonts.pdf", dpi=600, bbox_inches="tight")
plt.show()
