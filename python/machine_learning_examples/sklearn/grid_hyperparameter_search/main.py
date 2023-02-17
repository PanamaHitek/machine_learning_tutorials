from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

trainingSamples = 50000  # Self explanatory

"""
Here I set the global variables, which
will be used to simple_neural_network the computing time
for both training and testing
"""
def loadDataset(fileName, samples):  # A function for loading the data from a dataset
    x = []  # Array for data inputs
    y = []  # Array for labels (expected outputs)
    train_data = pd.read_csv(fileName)  # Data has to be stored in a CSV file, separated by commas
    y = np.array(train_data.iloc[0:samples, 0])  # Labels column
    x = np.array(train_data.iloc[0:samples, 1:]) / 255  # Division by 255 is used for data normalization
    return x, y

# Define the grid of hyperparameter values to search over
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Define the scoring metric to use for evaluation
scoring = 'accuracy'

# Create a GridSearchCV object
clf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=scoring, cv=3, verbose=2)

train_x,train_y = loadDataset("../../../../datasets/mnist/mnist_train.csv",10000)
test_x,test_y = loadDataset("../../../../datasets/mnist/mnist_test.csv",10000)

# Fit the grid search object to the training data
grid_search.fit(train_x, train_y)

# Print the best hyperparameters and the best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Re-train the model with the best hyperparameters
best_clf = grid_search.best_estimator_
best_clf.fit(train_x, train_y)

# Test the model with the best hyperparameters on the testing data
accuracy = best_clf.score(test_x, test_y)
print("Testing accuracy:", accuracy)
