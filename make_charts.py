import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style='darkgrid', context='notebook', palette='pastel')

out_dir = 'C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/'

# 1. Scale Ablation (10x vs 40x)
plt.figure(figsize=(8, 5))
models = ['10x Scale', '40x Scale']
immune_auroc = [0.578, 0.718]
chronic_auroc = [0.725, 0.756]
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, immune_auroc, width, label='Immune (AUROC)', color='#3498db')
plt.bar(x + width/2, chronic_auroc, width, label='Chronic (AUROC)', color='#e74c3c')
plt.ylabel('AUROC Score')
plt.title('Scale Ablation: 10x vs 40x Performance')
plt.xticks(x, models)
plt.ylim(0.4, 0.8)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'img_scale_ablation.png'), dpi=150)
plt.close()

# 2. Patch Size OOD (224 vs 512)
plt.figure(figsize=(8, 5))
modes = ['Regression', 'CORAL', 'CORAL_MT']
qwk_224 = [0.011, -0.089, -0.102]
qwk_512 = [-0.027, -0.185, -0.120]
x = np.arange(len(modes))
plt.bar(x - width/2, qwk_224, width, label='224px (Native)', color='#2ecc71')
plt.bar(x + width/2, qwk_512, width, label='512px (OOD)', color='#e67e22')
plt.ylabel('QWK Score')
plt.title('Patch Size Ablation: OOD Phenomenon on 512px')
plt.xticks(x, modes)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'img_patch_ood.png'), dpi=150)
plt.close()

# 3. ATI Proxy vs Real ATI (MAE)
plt.figure(figsize=(8, 5))
labels = ['KDIGO Proxy', 'Real ATI (Adjudicated)']
mae = [27.52, 21.65]
rmse = [32.90, 29.06]
x = np.arange(len(labels))
plt.bar(x - width/2, mae, width, label='MAE (%pt)', color='#9b59b6')
plt.bar(x + width/2, rmse, width, label='RMSE (%pt)', color='#34495e')
plt.ylabel('Error Rate (%) - Lower is better')
plt.title('Target Label Accuracy: Proxy vs Real ATI')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'img_ati_real_vs_proxy.png'), dpi=150)
plt.close()

print('Images generated successfully.')
