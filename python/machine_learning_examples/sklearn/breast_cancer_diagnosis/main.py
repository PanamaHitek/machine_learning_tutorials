import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def loadDataset(fileName):
    data = pd.read_csv(fileName)
    y = np.array(data.iloc[:, 5])
    x = np.array(data.iloc[:, :5])
    return x, y

dataset_x, dataset_y = loadDataset("../../../../datasets/mammographic_mass/dataset.csv")
train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier()

# Train the model
model.fit(train_x, train_y)

# Make predictions on the test data
pred_y = model.predict(test_x)

# Evaluate the model
accuracy = accuracy_score(test_y, pred_y)
print('\nTest accuracy:', accuracy)
