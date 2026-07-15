import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs('static/results', exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

tasks = ['Immune', 'Chronic', 'Stage 3']

# 1. Encoder Comparison Data
encoder_labels = ['ResNet50', 'CTransPath', 'DINOv2']
encoder_data = np.array([
    [0.238, 0.586, 0.443], # ResNet
    [0.395, 0.430, 0.485], # CTransPath
    [0.507, 0.505, 0.551]  # DINOv2
])

# 2. Scale Comparison Data
scale_labels = ['10x', '40x', 'Multi-scale']
scale_data = np.array([
    [0.395, 0.430, 0.485], # 10x
    [0.528, 0.728, 0.683], # 40x
    [0.719, 0.614, 0.539]  # Multi
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(tasks))
width = 0.25

# Plot 1: Encoder
ax1.bar(x - width, encoder_data[0], width, label='ResNet50 (ImageNet)', color='#95a5a6')
ax1.bar(x, encoder_data[1], width, label='CTransPath (Pathology)', color='#e74c3c')
ax1.bar(x + width, encoder_data[2], width, label='DINOv2 (General Vision)', color='#3498db')

ax1.set_title('Encoder Comparison (AUROC)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

for i in range(len(tasks)):
    ax1.text(i - width, encoder_data[0][i] + 0.01, f'{encoder_data[0][i]:.3f}', ha='center', fontsize=9)
    ax1.text(i, encoder_data[1][i] + 0.01, f'{encoder_data[1][i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax1.text(i + width, encoder_data[2][i] + 0.01, f'{encoder_data[2][i]:.3f}', ha='center', fontsize=9)

# Plot 2: Scale
ax2.bar(x - width, scale_data[0], width, label='10x (Context)', color='#f39c12')
ax2.bar(x, scale_data[1], width, label='40x (Cellular)', color='#2ecc71')
ax2.bar(x + width, scale_data[2], width, label='Multi-scale (10x+40x)', color='#8e44ad')

ax2.set_title('Scale Comparison (AUROC)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(tasks, fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

for i in range(len(tasks)):
    ax2.text(i - width, scale_data[0][i] + 0.01, f'{scale_data[0][i]:.3f}', ha='center', fontsize=9)
    ax2.text(i, scale_data[1][i] + 0.01, f'{scale_data[1][i]:.3f}', ha='center', fontsize=9)
    ax2.text(i + width, scale_data[2][i] + 0.01, f'{scale_data[2][i]:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(pad=3.0)
plt.savefig('static/results/ablation_studies_v2.png', dpi=300, bbox_inches='tight')
plt.close()
 