#!usr/bin/python

import sys
import datetime as dt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as knc

def read_file(file_name):
    my_file = open(file_name, "r")
    features = []
    labels = []

    while True:
        my_string = my_file.readline().split()
        if len(my_string) == 0:
            return features,labels
        else:
            labels.append(my_string[0])
            features.append(list(map(float,my_string[1:len(my_string)])))
                   
def print_result(true_result,predict_result):
    print("Confusion Matrix:")
    print(confusion_matrix(true_result,predict_result))
    print("Result:",accuracy_score(true_result,predict_result))



def knn_function(trainX,trainY,testX,testY):
    knn = knc(n_neighbors=3)
    #print(trainY)
    knn.fit(trainX,trainY)
    result = knn.predict(testX)
    print_result(testY,result)

def svm_function(trainX,trainY,testX,testY):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    #clf = svm.SVC()
    #clf.fit(trainX,trainY)
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(trainX, trainY)
    result = clf.predict(testX)
    print_result(testY,result) 

def main(argv):
    """if(len(argv) != 4):
        print("Input Error:")
        print("Use Mode: ./classify <classifier> <train file> <teste file>")
        exit()
    """
    train_features, train_labels = read_file(argv[1])
    test_features, test_labels = read_file(argv[2])

    #min_max_scaler = preprocessing.MinMaxScaler()
    #train_features = min_max_scaler.fit_transform(train_features)
    #test_features = min_max_scaler.transform(test_features)
    #test_features = min_max_scaler.fit_transform(test_features)
    
    if(argv[3] == "knn"):
        knn_function(train_features,train_labels,test_features,test_labels)
    elif(argv[3] == "svm"):
        svm_function(train_features,train_labels,test_features,test_labels)
        

if __name__ == "__main__":
    init = dt.datetime.now()
    main(sys.argv)
    end = dt.datetime.now()
    print("Time:",end - init)