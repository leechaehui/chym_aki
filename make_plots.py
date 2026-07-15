import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs('static/results', exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# 1. QWK Forest Plot
labels = ['Inflammation', 'Atrophy', 'Fibrosis']
qwk = [0.331, 0.383, 0.400]
ci_lower = [0.108, 0.168, 0.198]
ci_upper = [0.548, 0.566, 0.571]
err_lower = [q - l for q, l in zip(qwk, ci_lower)]
err_upper = [u - q for u, q in zip(ci_upper, qwk)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(qwk, labels, xerr=[err_lower, err_upper], fmt='o', color='#2c3e50', ecolor='#e74c3c', elinewidth=3, capsize=6, capthick=2, markersize=10)
ax.set_xlim(0, 0.7)
ax.set_title('Banff Descriptor Prediction Performance (QWK)', fontsize=15, fontweight='bold', pad=35)
ax.set_xlabel('Quadratic Weighted Kappa (95% CI)', fontsize=12, fontweight='bold', labelpad=15)
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.grid(axis='y', visible=False)
for i, (q, l, u) in enumerate(zip(qwk, ci_lower, ci_upper)):
    ax.text(q, i + 0.18, f'{q:.3f} ({l:.3f}-{u:.3f})', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
plt.tight_layout(pad=3.0)
plt.savefig('static/results/qwk_plot.png', dpi=300)
plt.close()

# 2. Stain AUROC Heatmap
tasks = ['Immune', 'Chronic', 'Stage 3']
stains = ['H&E', 'PAS', 'MT']
data = np.array([[0.689, 0.593, 0.613], [0.732, 0.781, 0.619], [0.543, 0.581, 0.630]])

fig, ax = plt.subplots(figsize=(8, 6))
cmap = sns.color_palette('YlGnBu', as_cmap=True)
sns.heatmap(data, annot=True, fmt='.3f', cmap=cmap, xticklabels=stains, yticklabels=tasks, annot_kws={'size': 14, 'weight': 'bold'}, cbar_kws={'label': 'AUROC Score'}, ax=ax, linewidths=1, linecolor='white')

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] == np.max(data[i, :]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='#e74c3c', lw=4))

ax.set_title('Single Stain Upper-Bound (AUROC)', fontsize=16, fontweight='bold', pad=40)
ax.set_ylabel('Prediction Task', fontsize=12, fontweight='bold', labelpad=15)
ax.set_xlabel('Stain Modality', fontsize=12, fontweight='bold', labelpad=15)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout(pad=3.0)
plt.savefig('static/results/stain_auroc_plot.png', dpi=300)
plt.close()
