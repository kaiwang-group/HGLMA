import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

files = [
    "HGLMA_108BiGG.xlsx - Fold_0.csv",
    "HGLMA_108BiGG.xlsx - Fold_1.csv",
    "HGLMA_108BiGG.xlsx - Fold_2.csv",
    "HGLMA_108BiGG.xlsx - Fold_3.csv",
    "HGLMA_108BiGG.xlsx - Fold_4.csv"
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)


# Convert 'Acc' (percentage) to 'Accuracy' (0-1 scale)
df['Accuracy'] = df['Acc'] / 100


metrics = ['AUPRC', 'Recall', 'F1', 'Accuracy']
plot_data = df[metrics]


df_melted = plot_data.melt(var_name='Metric', value_name='Value')


plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")



sns.boxplot(x='Metric', y='Value', data=df_melted, order=metrics, width=0.5, showfliers=False)


plt.ylim(0.4, 1.0)
plt.ylabel('Performance')
plt.xlabel('')


plt.savefig('performance_boxplot_no_outliers.png', dpi=300, bbox_inches='tight')
plt.show()