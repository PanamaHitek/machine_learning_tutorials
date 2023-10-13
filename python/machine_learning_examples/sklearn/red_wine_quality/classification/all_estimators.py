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

# Get all classification estimators
estimators = all_estimators(type_filter='classifier')

results = {}  # Dictionary to store results

for name, ClassifierClass in estimators:
    try:
        # Create a classifier instance
        model = ClassifierClass()

        # Perform 10-fold cross-validation and compute the average accuracy
        accuracies = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        avg_accuracy = accuracies.mean()

        results[name] = avg_accuracy
        print(f"{name} Average Accuracy: {avg_accuracy:.4f}")

    except Exception as e:
        print(f"Issue with {name}")  # This will catch both errors and warnings

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(list(results.items()), columns=['Classifier', 'Avg Accuracy']).sort_values(by='Avg Accuracy', ascending=False)

# Print the top 10 performers without the index
print(results_df.head(10).to_string(index=False))
