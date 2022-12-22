import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName, header=None)
    y = np.array(train_data.iloc[0:samples, 0])
    x = np.array(train_data.iloc[0:samples, 1:])
    return x, y


x, y = loadDataset("../../../../datasets/custom/custom_mnist.csv", 10)
fig = plt.figure(figsize=(2, 5))

for i in range(len(x)):
    digit = x[i]
    digit_pixels = digit.reshape(28, 28)
    fig.add_subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(digit_pixels)
plt.show()
