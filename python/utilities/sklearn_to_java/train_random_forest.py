import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import m2cgen as m2c

# Define a function to load the dataset
def loadDataset(fileName):
    data = pd.read_csv(fileName)
    y = np.array(data.iloc[:, 5])
    x = np.array(data.iloc[:, :5])
    return x, y

# Load the dataset
dataset_x, dataset_y = loadDataset("../../../datasets/mammographic_mass/dataset.csv")

# Split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)

# Define the Logistic Regression model with the best hyperparameters
model = RandomForestClassifier()

# Train the model
model.fit(train_x, train_y)

# Make predictions on the test data
pred_y = model.predict(test_x)

# Evaluate the model
accuracy = accuracy_score(test_y, pred_y)
print('Test accuracy:', accuracy)

# Generate the logistic regression model code in Java
code = m2c.export_to_java(model)

# Save the logistic regression model code to a Java file
with open('trained_model_random_forest.java', 'w') as f:
    f.write(code)

# Print a confirmation message
print('Random Forest model code saved to trained_model_random_forest.java')
