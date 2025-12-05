import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create the DataFrame with your data and associated error values
data = {
    'Feature': ['Bond Length', 'Neighbor Distance', 'Cellular Geometry', 'Structural Metrics',
                'Ordering Parameters', 'RDF', 'Spacegroup', 'Pore Features'],
    'MAE': [0.750656, 0.711116, 0.748745, 0.751765, 0.773895, 0.595195, 0.791218, 0.754687],
    'R2':  [0.250348, 0.311870, 0.231621, 0.254168, 0.188034, 0.493621, 0.161368, 0.223706],
    'std_R2': [0.030198, 0.016038, 0.003488, 0.020519, 0.013279, 0.028706, 0.022463, 0.016183],
    'std_MAE': [0.004991, 0.011976, 0.007549, 0.010630, 0.013327, 0.012483, 0.019864, 0.004235]
}

df = pd.DataFrame(data)

# Sort the DataFrame by R2 in descending order so that the error bars match the sorted order
df_sorted = df.sort_values('R2', ascending=False)

# Prepare positions for the grouped bar plot
x = np.arange(len(df_sorted))
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(12, 6))

color1 = '#3D91FF'
color2 = '#FF513D'

# Create side-by-side bar plots with error bars
ax.bar(x - width/2, df_sorted['R2'], width, yerr=df_sorted['std_R2'], capsize=5, label='R²', color=color1, alpha=0.8)
ax.bar(x + width/2, df_sorted['MAE'], width, yerr=df_sorted['std_MAE'], capsize=5, label='MAE', color=color2, alpha=0.8)

ax.set_xlabel('Feature Group', fontsize=14)
ax.set_ylabel('Metric Value', fontsize=14)
ax.set_title('Performance Metrics for Different Feature Groups', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(df_sorted['Feature'], rotation=45, ha='right', fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# Example horizontal lines for combined features (if needed)
ax.axhline(0.578511, color=color1, linestyle='--', label='Combined Features R²')
ax.axhline(0.542525, color=color2, linestyle='--', label='Combined Features MAE')

ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.show()