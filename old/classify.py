#!usr/bin/python

import sys
import datetime as dt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

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

def knn_function(train_values,train_labels,test_values,test_labels):
    knn = knc(n_neighbors=3)
    #print(train_labels)
    knn.fit(train_values,train_labels)
    result = knn.predict(test_values)
    print_result(test_labels,result)

def svm_function(train_values,train_labels,test_values,test_labels):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)

    """print("===========================================")
    print("               Cross Validation            ")
    scores = cross_val_score(clf, train_values, train_labels, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print("===========================================")
    print("               Prediction                  ")
    predicted = cross_val_predict(clf, train_values, train_labels, cv=10)
    x = accuracy_score(train_labels, predicted)
    print(x) 
    """
    print("===========================================")
    print("               Pure Result                 ")
    clf.fit(train_values, train_labels)
    print(clf.get_params())
    result = clf.score(test_values, test_labels)
    print(result)
    #print_result(test_labels,result) 

def main(argv):
    if(len(argv) != 4):
        print("Input Error:")
        print("Use Mode: ./classify <classifier> <train file> <teste file>")
        exit()
    
    train_features, train_labels = read_file(argv[2])
    test_features, test_labels = read_file(argv[3])

    #min_max_scaler = preprocessing.MinMaxScaler()
    #train_features = min_max_scaler.fit_transform(train_features)
    #test_features = min_max_scaler.transform(test_features)
    #test_features = min_max_scaler.fit_transform(test_features)
    
    if(argv[1] == "knn"):
        knn_function(train_features,train_labels,test_features,test_labels)
    elif(argv[1] == "svm"):
        svm_function(train_features,train_labels,test_features,test_labels)
        

if __name__ == "__main__":
    init = dt.datetime.now()
    main(sys.argv)
    end = dt.datetime.now()
    print("Time:",end - init)