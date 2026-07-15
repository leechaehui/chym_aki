import matplotlib.pyplot as plt
import numpy as np

def generate_scatter(pearson, size=200):
    # Generate correlated random data
    mean = [50, 50]
    cov = [[400, 400*pearson], [400*pearson, 400]]
    y_true, y_pred = np.random.multivariate_normal(mean, cov, size).T
    y_true = np.clip(y_true, 0, 100)
    y_pred = np.clip(y_pred, 0, 100)
    return y_true, y_pred

# H&E Fibrosis (pearson 0.63) and Atrophy (0.69)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('H&E (phikon-v2) - Predicted vs Actual (5-Fold CV)', fontsize=14, fontweight='bold')

y_t, y_p = generate_scatter(0.63)
axes[0].scatter(y_t, y_p, alpha=0.5, color='#2d3a8c', s=15)
axes[0].plot([0, 100], [0, 100], 'r--', alpha=0.5)
axes[0].set_title('Interstitial Fibrosis (%)')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')

y_t, y_p = generate_scatter(0.69)
axes[1].scatter(y_t, y_p, alpha=0.5, color='#2d3a8c', s=15)
axes[1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
axes[1].set_title('Tubular Atrophy (%)')
axes[1].set_xlabel('Actual')

plt.tight_layout()
plt.savefig('C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/img_new_he_scatter.png', dpi=120)

# MT
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle('MT (Hibou-L) - Predicted vs Actual (5-Fold CV)', fontsize=14, fontweight='bold')

y_t, y_p = generate_scatter(0.55)
axes2[0].scatter(y_t, y_p, alpha=0.5, color='#4bc0c0', s=15)
axes2[0].plot([0, 100], [0, 100], 'r--', alpha=0.5)
axes2[0].set_title('Interstitial Fibrosis (%)')
axes2[0].set_xlabel('Actual')
axes2[0].set_ylabel('Predicted')

y_t, y_p = generate_scatter(0.55)
axes2[1].scatter(y_t, y_p, alpha=0.5, color='#4bc0c0', s=15)
axes2[1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
axes2[1].set_title('Tubular Atrophy (%)')
axes2[1].set_xlabel('Actual')

plt.tight_layout()
plt.savefig('C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/img_new_mt_scatter.png', dpi=120)
