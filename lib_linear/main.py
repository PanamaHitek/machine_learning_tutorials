import csv
from sklearn import svm
import time

from sklearn.linear_model import LogisticRegression


def main():

    resultColumn  =0
    minDataColumn = 1
    maxDataColumn = 785
    x=[]
    y=[]
    resultCount=0
    validResults=0
    trainingSize = 50000
    testingSize = 10000
    trainingCounter =0
    testingCounter=0
    startTime=0
    endTime=0

    clf = None
    print("Training...")
    with open('../datasets/mnist/mnist_train.csv') as csvfile :
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            temp = []
            for i in range(minDataColumn,maxDataColumn):
                num = int(row[i])/255
                temp.append(num)
            x.append(temp)
            y.append(row[resultColumn])
            if (trainingCounter >= trainingSize):
                break
            else:
                trainingCounter = trainingCounter + 1
        clf = LogisticRegression(solver='liblinear')
        startTime=time.time()
        clf.fit(x, y)
        endTime = time.time()
    data = []
    print("Testing...")
    with open('../datasets/mnist/mnist_test.csv') as testfile:
        reader = csv.reader(testfile)
        next(reader)
        for row in reader:
            temp = []
            res = []
            for i in range(minDataColumn, maxDataColumn):
                num = int(row[i]) / 255
                temp.append(num)
            expectedResult = row[resultColumn]
            result = clf.predict([temp])
            res.append(row[resultColumn])
            res.append(result)
            data.append(res)
            resultCount=resultCount+1
            outcome = "Fail"
            if expectedResult==result:
                validResults=validResults+1
                outcome = "OK"
            print("Test N: ",resultCount," | Expected result: ",expectedResult," | Obtained result: ",result, " | Outcome: ",outcome ," | Accuracy: ",(validResults/resultCount)*100,"%");
            if (testingCounter >= testingSize-1):
                break
            else:
                testingCounter = testingCounter + 1

    print("--------------------")
    print("Results")
    print("--------------------")
    print("Performed tests: ",resultCount)
    print("Valid results: ", validResults)
    print("Accuracy: ",(validResults/resultCount)*100,"%")
    print("Training time: ", endTime-startTime)

if __name__ == "__main__":
    main()


