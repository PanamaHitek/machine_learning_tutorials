import pandas as pd
import numpy as np
import time as time
import tensorflow as tf

trainingSamples = 50000  # Number of samples used for training the model
testingSamples = 10000  # Number of samples used for testing the model

"""
Here I set the global variables, which
will be used to measure the computing time
for both training and testing
"""

startTrainingTime = 0
endTrainingTime = 0
trainingTime = 0

startTestingTime = 0
endTestingTime = 0
testingTime = 0

def loadDataset(fileName, samples):
    """
    A function for loading the data from a dataset.
    """
    x = []  # Array for data inputs
    y = []  # Array for labels (expected outputs)
    train_data = pd.read_csv(fileName)  # Data has to be stored in a CSV file, separated by commas
    y = np.array(train_data.iloc[0:samples, 0])  # Labels column
    x = np.array(train_data.iloc[0:samples, 1:]) / 255  # Division by 255 is used for data normalization
    return x, y

def main():
    """
    This function loads the data, defines the neural network architecture, trains and tests the model,
    and prints the results.
    """
    # Loading the training and testing data
    train_x, train_y = loadDataset("../../../../datasets/mnist/mnist_train.csv", trainingSamples)
    test_x, test_y = loadDataset("../../../../datasets/mnist/mnist_test.csv", testingSamples)

    # Defining the neural network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(784,)),  # Input layer: flatten the input images
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer: 128 neurons, ReLU activation
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer: 10 neurons (one for each digit), softmax activation
    ])

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    startTrainingTime = time.time()
    model.fit(train_x, train_y, epochs=10)  # Training of a model by fitting training data to object
    endTrainingTime = time.time()
    trainingTime = endTrainingTime - startTrainingTime  # Training time calculation

    # Testing the model
    validResults = 0
    startTestingTime = time.time()
    predictions = model.predict(test_x)
    for i in range(len(test_y)):
        # Evaluating the results
        expectedResult = int(test_y[int(i)])  # Load expected result from testing dataset
        result = np.argmax(predictions[i])  # Calculate a result using trained model
        outcome = "Fail"
        if result == expectedResult:
            validResults = validResults + 1  # Counting valid results
            outcome = " OK "
        print("Nº ", i + 1, " | Expected result: ", expectedResult, " | Obtained result: ", result, " | ", outcome,
              " | Accuracy: ", round((validResults / (i + 1)) * 100, 2),
              "%")  # Printing the results for each label in testing dataset

    endTestingTime = time.time()
    testingTime = endTestingTime - startTestingTime  # Calculation of testing time

    # Printing the results
    print("-------------------------------")
    print("Results")
    print("-------------------------------")
    print("Training samples: ", trainingSamples)
    print("Training time: ", round(trainingTime, 2), " s")
    print("Testing samples: ", testingSamples)
    print("Testing time: ", round(testingTime, 2), " s")
    print("Testing accuracy: ", round((validResults / testingSamples) * 100, 2), "%")

if __name__ == "__main__":
    main()
