import matplotlib.pyplot as plt
import numpy as np

# Use Times/serif font
plt.rcParams["font.family"] = "serif"

# Data
models = [
    "HAT", "RCAN", "EDDSR", "ArSSR",
    "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"
]

facts_synth = [11.66, 10.66, 6.12, 7.74, 11.49, 12.37, 13.69, 24.98, 27.30]
hcp_1200    = [10.02, 8.91, 6.08, 7.24, 7.22, 7.50, 12.93, 10.33, 12.93]

# Find best (min) and second-best for each dataset
def get_best_indices(values):
    sorted_idx = np.argsort(values)  # ascending order
    return sorted_idx[-1], sorted_idx[-2]

best_hcp, second_hcp = get_best_indices(hcp_1200)
best_facts, second_facts = get_best_indices(facts_synth)


# Bar positions
x = np.arange(len(models))
width = 0.35
gap = 0.15  # separation between bar pairs

# Colors (HCP-1200 first, then FACTS-Synth)
colors = ["burlywood", "darkseagreen"]

# Plot
fig, ax = plt.subplots(figsize=(11, 7))
bars1 = ax.bar(x - width/2 - gap/2, hcp_1200, width, label="HCP-1200",
               color=colors[0], edgecolor="black", linewidth=1.0)
bars2 = ax.bar(x + width/2 + gap/2, facts_synth, width, label="FACTS-Synth",
               color=colors[1], edgecolor="black", linewidth=1.0)

# Labels and styling
ax.set_ylabel("DI average", fontsize=22)
#ax.set_xlabel("Model", fontsize=15)
ax.set_title("Mean Diffusion Index averages", fontsize=25, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=20)
ax.tick_params(axis="y", labelsize=20)
ax.set_ylim(0, 30)

# Add grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

# Legend at bottom with larger font
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=22)

# Function to underline text using Unicode combining underline
def underline_text(s):
    return ''.join([c + '\u0332' for c in s])

# Annotate bars with best/second-best highlighted
def annotate_bars(bars, best_idx, second_idx):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == best_idx:
            text = f"$\\mathbf{{{height:.2f}}}$"   # bold
        elif i == second_idx:
            text = underline_text(f"{height:.2f}")  # underline
        else:
            text = f"{height:.2f}"
        ax.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=11)

#annotate_bars(bars1, best_hcp, second_hcp)       # HCP-1200
#annotate_bars(bars2, best_facts, second_facts)   # FACTS-Synth

plt.tight_layout()
plt.savefig("DI_barplot.pdf", dpi=600, bbox_inches="tight")
plt.show()
