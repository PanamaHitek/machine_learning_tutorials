import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators

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

# Define the Logistic Regression model with the best hyperparameters
model = LogisticRegression(C=0.1, max_iter=1000)

# Train the model
model.fit(train_x, train_y)

# Make predictions on the simple_neural_network data
pred_y = model.predict(test_x)

# Evaluate the model
accuracy = accuracy_score(test_y, pred_y)
print('Test accuracy:', accuracy)

# Print the coefficients and feature names
coefficients = model.coef_[0]
feature_names = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density']
for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
    print('Feature {}: {}, Coefficient: {:.3f}'.format(i, name, coef))
