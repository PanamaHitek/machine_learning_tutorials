# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')

# Visualize Distribution of Target Variable 'quality' using a histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(df['quality'], bins=range(1, 11), align='left', rwidth=0.8, color='skyblue', edgecolor='black')

# Add labels on top of each bar
for count, bin, patch in zip(counts, bins, patches):
    height = patch.get_height()
    plt.annotate(f'{int(count)}', xy=(bin, height), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom')

plt.title('Distribution of Wine Quality')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.xticks(range(1, 11))
plt.grid(axis='y')
plt.show()
