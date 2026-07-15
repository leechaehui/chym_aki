import matplotlib.pyplot as plt
import numpy as np

# Data
targets_he = ['Fibrosis', 'Atrophy', 'Tubular Injury', 'Inflammation', 'Arteriolar Hyal.']
mae_he = [16.8, 12.8, 18.6, 15.5, 31.3]
pearson_he = [0.63, 0.69, 0.25, 0.52, 0.38]
r2_he = [0.31, 0.45, -0.24, 0.23, -0.03]

targets_mt = ['Fibrosis', 'Atrophy', 'Arteriolar Hyal.']
mae_mt = [17.8, 15.7, 35.6]
pearson_mt = [0.55, 0.55, 0.33]
r2_mt = [0.20, 0.23, -0.14]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('H&E & MT - 5-Fold CV Metrics Summary', fontsize=16, fontweight='bold')

# MAE
x_he = np.arange(len(targets_he))
x_mt = np.arange(len(targets_mt))
axes[0].bar(x_he - 0.2, mae_he, width=0.4, label='H&E (phikon-v2)', color='#2d3a8c')
axes[0].bar(x_mt[:2] + 0.2, mae_mt[:2], width=0.4, label='MT (Hibou-L)', color='#4bc0c0')
axes[0].bar(4 + 0.2, mae_mt[2], width=0.4, color='#4bc0c0')
axes[0].set_xticks(x_he)
axes[0].set_xticklabels(targets_he, rotation=45, ha='right')
axes[0].set_title('MAE (%) - Lower is Better')
axes[0].legend()

# Pearson r
axes[1].bar(x_he - 0.2, pearson_he, width=0.4, color='#2d3a8c')
axes[1].bar(x_mt[:2] + 0.2, pearson_mt[:2], width=0.4, color='#4bc0c0')
axes[1].bar(4 + 0.2, pearson_mt[2], width=0.4, color='#4bc0c0')
axes[1].set_xticks(x_he)
axes[1].set_xticklabels(targets_he, rotation=45, ha='right')
axes[1].set_title('Pearson r - Higher is Better')

# R2
axes[2].bar(x_he - 0.2, r2_he, width=0.4, color='#2d3a8c')
axes[2].bar(x_mt[:2] + 0.2, r2_mt[:2], width=0.4, color='#4bc0c0')
axes[2].bar(4 + 0.2, r2_mt[2], width=0.4, color='#4bc0c0')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_xticks(x_he)
axes[2].set_xticklabels(targets_he, rotation=45, ha='right')
axes[2].set_title('R² - Higher is Better')

plt.tight_layout()
plt.savefig('C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/img_new_metrics_summary.png', dpi=150)
