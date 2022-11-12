import pandas as pd
import numpy as np

def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName)
    y = np.array(train_data.iloc[0:samples, 0])
    x = np.array(train_data.iloc[0:samples, 1:])
    return x,y

x,y=loadDataset("../datasets/mnist/mnist_train.csv",100)
print(x.shape)
print(y.shape)
