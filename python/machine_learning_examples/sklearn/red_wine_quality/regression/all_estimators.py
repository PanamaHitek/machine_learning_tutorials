import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils import all_estimators
import warnings

# Convert warnings to errors
warnings.simplefilter('error')  # This line will treat warnings as errors

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')

# Check for missing values and handle them
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean. Adjust this as needed.

# Split the data into features (X) and target variable (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Get all regression estimators
estimators = all_estimators(type_filter='regressor')

results = {}  # Dictionary to store results

for name, RegressorClass in estimators:
    try:
        # Create a regressor instance
        model = RegressorClass()

        # Perform 10-fold cross-validation and compute the average RMSE
        negative_mses = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
        avg_rmse = np.sqrt(-negative_mses.mean())

        results[name] = avg_rmse
        print(f"{name} Average RMSE: {avg_rmse}")

    except Exception as e:
        print(f"Issue with {name}")  # This will catch both errors and warnings

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(list(results.items()), columns=['Regressor', 'Avg RMSE']).sort_values(by='Avg RMSE')

# Print the top 10 performers without the index
print(results_df.head(10).to_string(index=False))
