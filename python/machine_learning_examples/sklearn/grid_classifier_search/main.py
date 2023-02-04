from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import pandas as pd
import numpy as np
import time as time

def loadDataset(fileName, samples):  # A function for loading the data from a dataset
    x = []  # Array for data inputs
    y = []  # Array for labels (expected outputs)
    train_data = pd.read_csv(fileName)  # Data has to be stored in a CSV file, separated by commas
    y = np.array(train_data.iloc[0:samples, 0])  # Labels column
    x = np.array(train_data.iloc[0:samples, 1:]) / 255  # Division by 255 is used for data normalization
    return x, y

train_x,train_y = loadDataset("../../../../datasets/mnist/mnist_train.csv",50000)
test_x,test_y = loadDataset("../../../../datasets/mnist/mnist_test.csv",10000)
# Create a dictionary to store the training accuracy of each classifier
accuracies = {}

# Create a list of classifiers
classifiers = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    BernoulliNB(),
    ComplementNB(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    MultinomialNB(),
    NearestCentroid(),
    NuSVC(),
    Perceptron(),
    RandomForestClassifier(),
    RidgeClassifier(),
    SGDClassifier(),
    SVC()
]

# Iterate over the classifiers and fit each one to the training data
for clf in classifiers:
    start = time.time()
    clf.fit(train_x, train_y)
    end = time.time()
    trainingTime = end - start
    start = time.time()
    accuracy = clf.score(test_x, test_y)
    end = time.time()
    testingTime = end - start
    accuracies[clf.__class__.__name__] = accuracy
    print(f"{clf.__class__.__name__} | training time: {trainingTime:.2f} seconds, testing time: {testingTime:.2f} seconds, {accuracy * 100:.2f}%")

# Find the classifier with the highest accuracy
best_classifier = max(accuracies, key=accuracies.get)
print("-----------------------------------")
print(f"Best classifier: {best_classifier}")