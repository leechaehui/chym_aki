import matplotlib.pyplot as plt
import numpy as np

# Data
models_he = ['phikon-v2', 'Hibou-L', 'DINOv2', 'DINO(ep200)']
mae_he = [19.0, 22.5, 21.0, 24.5]

models_mt = ['Hibou-L', 'phikon-v2', 'DINOv2', 'DINO(ep200)']
mae_mt = [19.5, 23.0, 35.0, 25.5]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Model Comparison (512x, ABMIL 5-Fold CV) - Average MAE', fontsize=14, fontweight='bold')

axes[0].bar(models_he, mae_he, color=['#2d3a8c', '#cccccc', '#cccccc', '#cccccc'])
axes[0].set_title('H&E Stain')
axes[0].set_ylabel('Mean Absolute Error (%)')
for i, v in enumerate(mae_he):
    axes[0].text(i, v + 0.5, str(v), ha='center')

axes[1].bar(models_mt, mae_mt, color=['#4bc0c0', '#cccccc', '#cccccc', '#cccccc'])
axes[1].set_title("Masson's Trichrome Stain")
for i, v in enumerate(mae_mt):
    axes[1].text(i, v + 0.5, str(v), ha='center')

plt.tight_layout()
plt.savefig('C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/img_new_model_comparison.png', dpi=120)
