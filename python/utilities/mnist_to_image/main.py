import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName)
    y = np.array(train_data.iloc[0:samples, 0])
    x = np.array(train_data.iloc[0:samples, 1:])
    return x, y


x, y = loadDataset("../../../datasets/mnist/mnist_test.csv", 10000)

digit = x[1327]
digit_pixels = digit.reshape(28, 28)
plt.imshow(digit_pixels)
plt.show()
