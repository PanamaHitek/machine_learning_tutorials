import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import m2cgen as m2c
import time

trainingSamples = 50000

def loadDataset(fileName, samples):
    # Load a dataset from a CSV file, and return the features and labels as NumPy arrays
    train_data = pd.read_csv(fileName)
    y = np.array(train_data.iloc[0:samples, 0])  # Extract the labels from the first column
    x = np.array(train_data.iloc[0:samples, 1:]) / 255  # Extract the features, and normalize them by dividing by 255
    return x, y

def main():
    # Load the MNIST training dataset and the number of samples to use for training
    train_x, train_y = loadDataset("../../../datasets/mnist/mnist_train.csv", trainingSamples)

    # Train the random forest classifier and time the training process
    startTrainingTime = time.time()
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    endTrainingTime = time.time()
    trainingTime = endTrainingTime - startTrainingTime

    # Export the trained model to Java code and time the process
    startExportTime = time.time()
    java_code = m2c.export_to_python(clf)
    endExportTime = time.time()
    exportTime = endExportTime - startExportTime

    # Save the Java code to a text file, and print the training time and export time
    with open('trained_model.txt', 'w') as f:
        f.write(java_code)
    print("Trained model saved to trained_model.py")
    print("Training time: ", round(trainingTime, 2), " seconds")
    print("Export time: ", round(exportTime, 2), " seconds")

if __name__ == "__main__":
    main()
