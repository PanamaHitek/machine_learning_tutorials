# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')


def plot_all_features_boxplots():
    """
    Creates horizontal boxplots for all features in the dataset arranged in a grid.
    """

    features = df.columns.drop('quality')  # Assuming 'quality' is not a feature to be plotted
    n = len(features)

    n_rows = int(n / 4) + (n % 4 > 0)  # Calculate the number of rows needed for the grid
    plt.figure(figsize=(20, n_rows * 2))  # Adjusting figure size based on number of features

    for i, feature_name in enumerate(features):
        plt.subplot(n_rows, 4, i + 1)  # Arranging plots in grid
        sns.boxplot(x=df[feature_name], orient="h", color="skyblue", fliersize=5,
                    linewidth=1.5, width=0.7, flierprops=dict(markerfacecolor='r', marker='s'))
        plt.title(f'Boxplot of {feature_name}')
        plt.xlabel(feature_name)
        plt.grid(axis='x')

    plt.tight_layout()
    plt.show()


plot_all_features_boxplots()
