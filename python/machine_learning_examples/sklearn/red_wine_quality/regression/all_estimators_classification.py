import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings

# Convert warnings to errors
warnings.simplefilter('error')  # This line will treat warnings as errors

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')

# Check for missing values and handle them
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean. Adjust this as needed.

# Split the data into features (X) and target variable (y), and then split into training and test sets
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get all regression estimators
estimators = all_estimators(type_filter='regressor')

results = {}  # Dictionary to store accuracy results

for name, RegressorClass in estimators:
    try:
        # Create a regressor instance
        model = RegressorClass()
        model.fit(X_train, y_train)

        # Predict using the regressor and round the results
        predictions = model.predict(X_test)
        rounded_predictions = np.round(predictions)

        # Compute accuracy comparing rounded predictions with actual values
        accuracy = accuracy_score(y_test, rounded_predictions)

        results[name] = accuracy
        print(f"{name} Accuracy (using rounded predictions): {accuracy:.4f}")

    except Exception as e:
        print(f"Issue with {name}")  # This will catch both errors and warnings

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(list(results.items()), columns=['Regressor', 'Accuracy']).sort_values(by='Accuracy',
                                                                                                ascending=False)
# Print the top 10 performers without the index
print(results_df.head(10).to_string(index=False))
