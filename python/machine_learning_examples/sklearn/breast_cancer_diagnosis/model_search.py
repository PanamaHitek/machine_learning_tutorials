import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Get a list of all classification models in scikit-learn
estimators = all_estimators(type_filter='classifier')

# Define variables to store the best model and its accuracy
best_model = None
best_accuracy = -1

# Loop over all classification models
for name, ClassifierClass in estimators:
    try:
        # Create an instance of the model
        model = ClassifierClass()

        # Train the model
        model.fit(train_x, train_y)

        # Make predictions on the simple_neural_network data
        pred_y = model.predict(test_x)

        # Evaluate the model
        accuracy = accuracy_score(test_y, pred_y)

        # Update the best model and its accuracy
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

        print('Model: {}, Accuracy: {}'.format(name, accuracy))

    except Exception as e:
        # Some models may raise exceptions, so we catch them and print an error message
        print('Error with model {}: {}'.format(name, e))

# Print the best model and its accuracy
print('\nBest model: {}, Accuracy: {}'.format(type(best_model).__name__, best_accuracy))
