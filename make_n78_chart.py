import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
n78 = pd.read_csv('C:/team/chym_aki/data/eval/attention/ati_exp_compare_N78.csv')
n83 = pd.read_csv('C:/team/chym_aki/data/eval/attention/ati_exp_compare.csv')

modes = ['regression', 'coral', 'coral_mt']
qwk_78 = n78['QWK_5class'].values
qwk_83 = n83['QWK_5class'].values

x = np.arange(len(modes))
width = 0.35

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, qwk_78, width, label='N=78 (Previous)', color='#5e81ac')
rects2 = ax.bar(x + width/2, qwk_83, width, label='N=83 (Current)', color='#ebcb8b')

ax.set_ylabel('QWK (Quadratic Weighted Kappa)')
ax.set_title('Impact of Adding Patients (N=78 vs N=83) on CDSS Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(['Regression', 'CORAL', 'CORAL + MT Aux'])
ax.legend()

# Add labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('C:/Users/301-4/.gemini/antigravity-ide/brain/7a6536d9-bcd6-4caa-ba4b-d3c75eaa3491/img_n78_vs_n83_comparison.png', dpi=150)
