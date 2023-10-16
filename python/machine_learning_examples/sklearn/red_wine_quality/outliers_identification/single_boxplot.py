# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')

def plot_boxplot(feature_name):
    """
    Creates a horizontal boxplot for the given feature name.

    Parameters:
    - feature_name: Name of the feature for which the boxplot is to be created.
    """

    # Check if the feature exists in the dataset
    if feature_name in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[feature_name], orient="h")  # Made the boxplot horizontal
        plt.title(f'Boxplot of {feature_name}')
        plt.xlabel(feature_name)  # Changed ylabel to xlabel
        plt.grid(axis='x')  # Changed axis to 'x'
        plt.tight_layout()
        plt.show()
    else:
        print(f"Feature '{feature_name}' not found in the dataset.")

# Call the function with desired feature name
plot_boxplot('pH')  # You can replace 'alcohol' with any other feature name
