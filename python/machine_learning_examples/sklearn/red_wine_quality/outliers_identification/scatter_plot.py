# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')

# Drop the 'quality' column as we are focusing on features
features = df.drop('quality', axis=1)

# Set up the figure and axes
fig, axs = plt.subplots(4, 3, figsize=(16, 8))  # Increased figsize for better visibility

# Flatten the axes array for easier indexing
axs = axs.ravel()

# Create scatter plots of each feature against 'quality'
for i, column in enumerate(features.columns):
    axs[i].scatter(df[column], df['quality'], alpha=0.5, edgecolor='black')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Wine Quality')
    axs[i].grid(True)

# Remove any unused subplots
for j in range(i + 1, 12):  # 12 is the total number of subplots (4x3)
    fig.delaxes(axs[j])

plt.subplots_adjust(hspace=0.5)  # Adjusted vertical space between plots
plt.tight_layout()
plt.show()
