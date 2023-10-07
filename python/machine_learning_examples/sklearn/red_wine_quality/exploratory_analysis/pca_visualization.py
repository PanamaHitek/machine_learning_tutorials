# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')  # Replace 'path_to_your_dataset.csv' with your dataset path

# Separate features and target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to three principal components
pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(X_scaled)

# Convert to DataFrame for easier plotting
pca_df_3d = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['quality'] = y

# Plot in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for quality in sorted(pca_df_3d['quality'].unique()):
    subset = pca_df_3d[pca_df_3d['quality'] == quality]
    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], label=f'Quality {quality}', alpha=0.7)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Red Wine Quality Dataset')
ax.legend()
plt.show()
