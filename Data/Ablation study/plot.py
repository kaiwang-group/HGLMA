import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Define the file mapping
files = {
    "HGLMA": "HGLMA_ablation.xlsx - HGLMA.csv",
    "HGLMA-v1": "HGLMA_ablation.xlsx - HGLMA-v1.csv",
    "HGLMA-v2": "HGLMA_ablation.xlsx - HGLMA-v2.csv",
    "HGLMA-v3": "HGLMA_ablation.xlsx - HGLMA-v3.csv"
}

data_frames = []

# Loop through each file to load and clean data
for label, filename in files.items():
    try:
        # Read raw lines to handle potential double-quote formatting issues
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Remove whitespace and outer quotes from each line
        cleaned_lines = [line.strip().strip('"') for line in lines]

        # Create DataFrame from cleaned string data
        df = pd.read_csv(io.StringIO('\n'.join(cleaned_lines)))

        # Add a column to identify the method/version
        df['Method'] = label
        data_frames.append(df)

    except FileNotFoundError:
        print(f"File not found: {filename}")

# Combine all dataframes
if data_frames:
    full_df = pd.concat(data_frames, ignore_index=True)

    # Normalize 'Acc' (Accuracy) to 0-1 range for consistent plotting
    # Assuming Acc is 0-100 and others are 0-1
    full_df['Acc'] = full_df['Acc'] / 100.0

    # Melt the DataFrame to long format for Seaborn
    melted_df = full_df.melt(id_vars='Method', var_name='Metric', value_name='Score')

    # Set up the plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Create the violin plot
    # inner='quartile' draws the quartile lines inside the violins
    sns.violinplot(
        data=melted_df,
        x='Metric',
        y='Score',
        hue='Method',
        inner='quartile',
        palette='muted'
    )

    # Add labels and title
    plt.title('Performance Distribution by Metric and Method', fontsize=16)
    plt.ylabel('Score (Acc scaled to 0-1)', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.legend(title='Method', loc='lower right')

    # Save and show the plot
    plt.savefig('violin_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data loaded.")