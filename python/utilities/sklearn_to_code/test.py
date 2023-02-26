import trained_model  # Import the trained_model module containing the trained model
import pandas as pd  # Import the Pandas library for working with data
import numpy as np  # Import the NumPy library for working with arrays
import time  # Import the time library for measuring the testing time

trainingSamples = 50000  # The number of training samples used for training the model
testingSamples = 10000  # The number of testing samples to evaluate the model

def loadDataset(fileName, samples):
    """
    Load a dataset from a CSV file.

    Args:
        fileName: The path to the CSV file.
        samples: The number of samples to load.

    Returns:
        A tuple of two NumPy arrays: one containing the input data (x) and the other containing the output labels (y).
    """
    x = []
    y = []

    train_data = pd.read_csv(fileName)  # Load the data from the CSV file using Pandas
    y = np.array(train_data.iloc[0:samples, 0])  # Get the labels (first column) for the specified number of samples
    x = np.array(train_data.iloc[0:samples, 1:]) / 255  # Get the input data (remaining columns) and normalize it

    return x, y

# Load the testing dataset
test_x, test_y = loadDataset("../../../datasets/mnist/mnist_test.csv", testingSamples)

validResults = 0  # The number of correctly classified samples
startTestingTime = time.time()  # Get the current time to measure the testing time

# Iterate over the testing data and classify each sample
for i in range(len(test_y)):
    # Get the expected result from the dataset
    expectedResult = int(test_y[int(i)])

    # Use the trained model to get a confidence score for each possible class
    score = trained_model.score(test_x[int(i)])

    # Find the class with the highest confidence score and use it as the predicted result
    result = score.index(max(score))

    # Check if the predicted result matches the expected result
    outcome = ""  # Initialize the outcome to an empty string
    if result == expectedResult:
        validResults += 1
        outcome = " OK "

    # Print the classification results for each sample
    print("Nº ", i + 1, " | Expected result: ", expectedResult, " | Obtained result: ", result, " | ", outcome,
          " | Accuracy: ", round((validResults / (i + 1)) * 100, 2), "%")

endTestingTime = time.time()
testingTime = endTestingTime - startTestingTime

# Print the final testing results
print("-------------------------------")
print("Results")
print("-------------------------------")
print("Testing samples: ", testingSamples)
print("Testing time: ", round(testingTime, 2), " s")
print("Testing accuracy: ", round((validResults / testingSamples) * 100, 2), "%")
