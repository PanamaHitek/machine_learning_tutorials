# Import necessary libraries
import pandas as pd
import numpy as np
import time as time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the number of training and testing samples
trainingSamples = 50000
testingSamples = 10000

"""
Set global variables for the computing time of
both training and testing
"""
startTrainingTime = 0
endTrainingTime = 0
trainingTime = 0

startTestingTime = 0
endTestingTime = 0
testingTime = 0


# Function for loading data from a CSV file and returning the data and labels
def loadDataset(fileName, samples):
    x = []  # Array for data inputs
    y = []  # Array for labels (expected outputs)

    # Load data from a CSV file and store it in a pandas DataFrame object
    with open(fileName, 'r') as f:
        train_data = pd.read_csv(f)

    # Extract labels from the first column of the DataFrame object
    y = np.array(train_data.iloc[0:samples, 0])

    # Extract data from the remaining columns of the DataFrame object and normalize it
    x = np.array(train_data.iloc[0:samples, 1:]) / 255

    # Convert the data and labels to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Return the data and labels
    return x, y


# Define a simple neural network class
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()

        # Define a fully connected layer with 784 input features (28x28 images) and 128 output features
        self.fc1 = nn.Linear(784, 128)

        # Define a fully connected layer with 128 input features and 10 output features (for the 10 digits)
        self.fc2 = nn.Linear(128, 10)

        # Define the activation function as ReLU
        self.activation = nn.ReLU()

    def forward(self, x):
        # Flatten the input image into a 1D vector
        x = x.view(-1, 784)

        # Pass the input through the first fully connected layer and apply the activation function
        x = self.fc1(x)
        x = self.activation(x)

        # Pass the output of the first layer through the second fully connected layer
        x = self.fc2(x)

        # Return the output
        return x


# Main function
def main():
    # Load the training and testing datasets
    train_x, train_y = loadDataset("../../../../datasets/mnist/mnist_train.csv", trainingSamples)
    test_x, test_y = loadDataset("../../../../datasets/mnist/mnist_test.csv", testingSamples)

    # Define the batch size for training and testing data
    batchSize = 64

    # Define a DataLoader object for the training data
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batchSize, shuffle=True)

    # Define a DataLoader object for the testing data
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batchSize)

    device = torch.device("cpu")  # remove to use GPU, is available
    # Create a SimpleNeuralNetwork object and define the loss function and optimizer
    model = SimpleNeuralNetwork().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model and measure the time
    startTrainingTime = time.time()

    # Set the number of epochs for training
    epochs = 20

    # Loop over the specified number of epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        # Loop over the batches of training data
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Zero the gradients of the model parameters
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_function(outputs, labels)

            # Backward pass through the model and update the parameters
            loss.backward()
            optimizer.step()

            # Update the epoch loss and accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            epoch_accuracy += correct / batchSize

        # Print out the epoch loss and accuracy
        print("Epoch:", epoch + 1, " Loss:", epoch_loss / len(train_loader), " Accuracy:",
              epoch_accuracy / len(train_loader))

    endTrainingTime = time.time()
    trainingTime = endTrainingTime - startTrainingTime

    # Evaluate the model on the testing data and measure the time
    validResults = 0
    startTestingTime = time.time()
    for i in range(len(test_y)):
        inputs = test_x[i].unsqueeze(0)
        labels = torch.tensor([test_y[i]], dtype=torch.long)

        # Forward pass through the model
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        result = predicted.item()
        expectedResult = int(labels[0])
        outcome = "Fail"
        if result == expectedResult:
            validResults = validResults + 1
            outcome = " OK "

        print("Nº ", i + 1, " | Expected result: ", expectedResult, " | Obtained result: ", result, " | ", outcome,
              " | Accuracy: ", round((validResults / (i + 1)) * 100, 2),
              "%")

    endTestingTime = time.time()
    testingTime = endTestingTime - startTestingTime

    # Calculate and print out the training and testing time and the testing accuracy
    print("-------------------------------")
    print("Results")
    print("-------------------------------")
    print("Training samples: ", trainingSamples)
    print("Training time: ", round(trainingTime, 2), " s")
    print("Testing samples: ", testingSamples)
    print("Testing time: ", round(testingTime, 2), " s")
    print("Testing accuracy: ", round((validResults / testingSamples) * 100, 2), "%")


# Run the main function if this file is being executed directly
if __name__ == "__main__":
    main()
