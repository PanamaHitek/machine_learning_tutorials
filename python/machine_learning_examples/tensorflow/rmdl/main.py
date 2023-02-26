import pandas as pd
import numpy as np
import time as time
from RMDL import RMDL
from keras.datasets import mnist

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


def loadDataset():
    """
    A function for loading the data from the MNIST dataset.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    return x_train[:trainingSamples], y_train[:trainingSamples], x_test[:testingSamples], y_test[:testingSamples]


def main():
    """
    This function loads the data, defines the RMDL model architecture, trains and tests the model,
    and prints the results.
    """
    # Loading the training and testing data
    x_train, y_train, x_test, y_test = loadDataset()

    # Defining the RMDL model architecture
    model = RMDL(image_rows=28, image_cols=28, image_channels=1, class_num=10, pretrained=False, feature_extract=True,
                 fine_tuning=True, use_imagenet=False, block_per_level=[2, 2], nb_epochs=[10, 10], batch_size=128,
                 random_state=42)

    # Training the model
    startTrainingTime = time.time()
    model.fit(x_train, y_train)
    endTrainingTime = time.time()
    trainingTime = endTrainingTime - startTrainingTime  # Training time calculation

    # Testing the model
    startTestingTime = time.time()
    _, accuracy = model.evaluate(x_test, y_test)
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
    print("Testing accuracy: ", round(accuracy * 100, 2), "%")


if __name__ == "__main__":
    main()
