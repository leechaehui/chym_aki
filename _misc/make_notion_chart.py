import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Data
targets = ['Fibrosis\nRatio', 'Atrophy\nRatio', 'Art.\nHyalinosis']
models = ['phikon-v2', 'DINOv2-L', 'Hibou-L']
colors = ['#5e81ac', '#d08770', '#a3be8c'] # Adjusted Notion-like colors. Wait, the image shows: Blue, Orange, Green.
colors = ['#5b84b1', '#dd8e67', '#6ab17d']

# Row 1: MAE
mae_data = [
    [18.3, 15.0, 38.0], # phikon
    [25.4, 22.6, 44.1], # dino
    [17.8, 15.7, 35.6]  # hibou
]
mae_gold = [2, 0, 2] # Index of model with gold border for each target

# Row 2: Pearson r
pearson_data = [
    [0.58, 0.61, 0.19],
    [0.04, -0.07, -0.05],
    [0.55, 0.55, 0.33]
]
pearson_gold = [0, 0, 2]

# Row 3: R2
r2_data = [
    [0.19, 0.27, -0.32],
    [-0.36, -0.43, -0.60],
    [0.20, 0.23, -0.14]
]
r2_gold = [2, 0, 2]

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle("Masson's Trichrome (MT) — Model Comparison (512x, ABMIL 5-Fold CV)", fontsize=14, fontweight='bold', y=0.98)

x = np.arange(len(targets))
width = 0.25

def plot_bar(ax, data, gold_indices, ylabel, ylim=None, show_legend=False):
    for m_idx in range(3):
        bars = ax.bar(x + (m_idx - 1)*width, data[m_idx], width, color=colors[m_idx], edgecolor='white', linewidth=1)
        for t_idx, bar in enumerate(bars):
            if gold_indices[t_idx] == m_idx:
                bar.set_edgecolor('#f4c430')
                bar.set_linewidth(2)
            
            # Add text labels
            height = bar.get_height()
            if height >= 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 if ylim and ylim[1]<=1 else 1),
                        f'{height:.1f}%' if '%' in ylabel or data[m_idx][t_idx] > 2 else f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9, color='#444444')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height - (0.02 if ylim and ylim[1]<=1 else 1),
                        f'{height:.2f}',
                        ha='center', va='top', fontsize=9, color='#444444')
    
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='gray', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ylim:
        ax.set_ylim(ylim)

plot_bar(axes[0], mae_data, mae_gold, 'MAE (%)', ylim=(0, 55))
plot_bar(axes[1], pearson_data, pearson_gold, 'Pearson r', ylim=(-0.1, 1.0))
plot_bar(axes[2], r2_data, r2_gold, 'R²', ylim=(-0.8, 1.0))

axes[2].set_xticks(x)
axes[2].set_xticklabels(targets)

# Legend
import matplotlib.lines as mlines
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='white', label='phikon-v2  (avg MAE 23.8%)'),
    Patch(facecolor=colors[1], edgecolor='white', label='DINOv2-L  (avg MAE 30.7%)'),
    Patch(facecolor=colors[2], edgecolor='white', label='Hibou-L  (avg MAE 23.0%)')
]
axes[0].legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
axes[0].text(0.01, 0.95, '★ gold border = best per target', transform=axes[0].transAxes, color='#d4a017', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/img_mt_comparison_notion.png', dpi=150)
