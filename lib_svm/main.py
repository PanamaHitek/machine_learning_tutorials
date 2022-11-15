import pandas as pd
import numpy as np
import sys
from libsvm.svmutil import svm_train, svm_predict


def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName)
    y = np.array(train_data.iloc[0:samples, 0])
    x = np.array(train_data.iloc[0:samples, 1:])/255
    return x,y

train_x,train_y=loadDataset("../datasets/mnist/mnist_train.csv",50000)
m = svm_train(train_y, train_x, '-s 0 -d 1 -g 0.025 -c 100 -t 2 -q')

test_x,test_y=loadDataset("../datasets/mnist/mnist_test.csv",10000)
p_label, p_acc, p_val = svm_predict(test_y, test_x, m,options="-q")
print(p_acc)