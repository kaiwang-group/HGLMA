import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "HGLMA_recovery.xlsx - Sheet1.csv"
df = pd.read_csv(file_path)


part1 = df[['Top25', 'Top50', 'Top100', 'TopN']].copy()

part2 = df[['Top25.1', 'Top50.1', 'Top100.1', 'TopN.1']].copy()
part2.columns = ['Top25', 'Top50', 'Top100', 'TopN']


combined_df = pd.concat([part1, part2], ignore_index=True)


plot_data = combined_df.melt(var_name='Metric', value_name='Value')


plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")


sns.boxplot(x='Metric', y='Value', data=plot_data,
            order=['Top25', 'Top50', 'Top100', 'TopN'],
            width=0.5, showfliers=False)


plt.ylim(0.4, 1.0)  # Range 0.4 - 1.0
plt.ylabel('Performance')
plt.xlabel('')
plt.title('HGLMA_recovery')


plt.savefig('recovery_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()