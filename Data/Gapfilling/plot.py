import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('xxxx.xlsx')

method_mapping = {
    'Draft': 'Draft',
}

df['method_label'] = df['method'].map(method_mapping)

metrics = ['roc_auc', 'prc_auc', 'recall', 'precision', 'f1', 'accuracy', 'matthews_corrcoef']

n_cols = 4
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
axes = axes.flatten()

method_order = ['Draft', 'NHP', 'Random', 'CHESHIRE']

for i, metric in enumerate(metrics):
    sns.boxplot(x='method_label', y=metric, data=df, ax=axes[i], order=method_order, palette='Set2')
    axes[i].set_title(metric)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

for i in range(len(metrics), len(axes)):
    fig.delaxes(axes[i])

plt.savefig('metrics_boxplots.png')
plt.show()