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


x, y = loadDataset("../datasets/mnist/mnist_train.csv", 50000)

fig = plt.figure(figsize=(10, 10))
limit = 100
limitIndex = 0
labelToPrint = 9

for i in range(len(x)):
    digit = x[i]
    label = int(y[i])
    if (label == labelToPrint):
        digit_pixels = digit.reshape(28, 28)
        fig.add_subplot(10, 10, limitIndex + 1)
        plt.axis('off')
        plt.imshow(digit_pixels)
        if (limitIndex >= limit - 1):
            break
        limitIndex = limitIndex + 1
plt.show()
