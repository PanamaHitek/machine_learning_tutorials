import pandas as pd

def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName)
    y = train_data.iloc[0:samples, 0]
    x = train_data.iloc[0:samples, 1:]
    return x,y

x,y=loadDataset("../datasets/mnist/mnist_train.csv",100)
print(x.shape)
print(y.shape)
