import pandas as pd
import numpy as np
from liblinear.liblinear import problem, parameter, liblinear


def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName)
    y = np.array(train_data.iloc[0:samples, 0])
    x = np.array(train_data.iloc[0:samples, 1:])/255
    return x,y

train_x,train_y=loadDataset("../datasets/mnist/mnist_train.csv",1000)
prob = problem(train_y, train_x)
param = parameter('-c 4')
m = liblinear.train(prob, param)

test_x,test_y=loadDataset("../datasets/mnist/mnist_test.csv",100)
p_labels, p_acc, p_vals = liblinear.predict(test_y, test_x, m)
print(p_acc)