import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define a function to load the dataset
def loadDataset(fileName):
    data = pd.read_csv(fileName)
    y = np.array(data.iloc[:, 5])
    x = np.array(data.iloc[:, :5])
    return x, y

# Load the dataset
dataset_x, dataset_y = loadDataset("../../../../datasets/mammographic_mass/dataset.csv")

# Split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier()

# Define the hyperparameters to search over
param_grid = {
    'n_estimators': [300, 500, 750],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2)
grid_search.fit(train_x, train_y)

# Make predictions on the simple_neural_network data using the best model found
best_model = grid_search.best_estimator_
pred_y = best_model.predict(test_x)

# Evaluate the model
accuracy = accuracy_score(test_y, pred_y)
print('\nTest accuracy:', accuracy)

# Print the best hyperparameters found
print('\nBest hyperparameters:', grid_search.best_params_)
