#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
from sklearn import svm
from sources.classes import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc


def readData(file_name):
    data = File(file_name, "r")
    data_features = []
    data_labels = []
    while True:
        x, tam = data.readList()
        #print(x,tam)
        if tam == 0:
            return data_features, data_labels
        else:
            data_labels.append(x[0])
            data_features.append(x[1:len(x)])

def printResult(labels, result):
    #log = open("log.txt","w+")
    print("Confusion Matrix: ")
    print(confusion_matrix(labels,result))
    print("Accuracy: ",accuracy_score(labels,result))


def knn_function(train_features, train_labels, test_features, test_labels):
    print("\n\nKNN")
    print("===================================================================")
    ks = [1, 3, 5]
    for k in ks:
        knn = knc(n_neighbors=k)
        knn.fit(train_features, train_labels)
        result = knn.predict(test_features)
        print("k = ",k)
        printResult(test_labels, result)
    print("===================================================================")


def svm_function(train_features, train_labels, test_features, test_labels):
    print("\n\nSVM")
    print("===================================================================")
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    #svr = svm.SVC(C=10, kernel='linear')
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters).fit(train_features, train_labels)
    # print(clf.best_params_)
    result = clf.predict(test_features)
    printResult(test_labels, result)
    print("===================================================================")


def mlp_function(train_features, train_labels, test_features, test_labels):
    print("\n\nMLP")
    print("===================================================================")
    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(train_features, train_labels)
    result = mlp.predict(test_features)
    printResult(test_labels, result)
    print("===================================================================")


def ensemble(train_features, train_labels, test_features, test_labels):
    print("\n\nEnsemble")
    print("===================================================================")
    ks = [1, 3, 5]
    for k in ks:
        print("k = ",k)
        m_knn = knc(n_neighbors=k)

        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svr = svm.SVC(probability=True)
        m_svm = GridSearchCV(svr, parameters).fit(train_features, train_labels)

        m_mlp = MLPClassifier(max_iter=1000)

        clf1 = VotingClassifier(estimators=[('knn',m_knn),('svm',m_svm),('mpl',m_mlp)], voting='hard')
        clf2 = VotingClassifier(estimators=[('knn',m_knn),('svm',m_svm),('mpl',m_mlp)], voting='soft')

        clf1.fit(train_features, train_labels)
        clf2.fit(train_features, train_labels)

        result1 = clf1.predict(test_features)
        result2 = clf2.predict(test_features)

        printResult(test_labels, result)
        printResult(test_labels, result)
    print("===================================================================")


def classify():

    data = ['features/lbp_default_train.txt',
            'features/lbp_ror_train.txt',
            'features/lbp_uniform_train.txt',
            'features/lbp_nri_uniform_train.txt',
            'features/haralick_train.txt',
            'features/glcm_1_train.txt',
            'features/glcm_2_train.txt',
            'features/glcm_3_train.txt',
            'features/glcm_4_train.txt',
            'features/glcm_5_train.txt']

    for d in data:
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(d)
        data_features, data_labels = readData(d)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_features = min_max_scaler.fit_transform(data_features)

        for i in range(0, 1):
            rand = random.randint(1, 100)
            train_f ,test_f,train_l,test_l = train_test_split(data_features,
                                                              data_labels,
                                                              test_size=0.4,
                                                              random_state=rand)

            mlp_function(train_f, train_l, test_f, test_l)
            knn_function(train_f, train_l, test_f, test_l)
            svm_function(train_f, train_l, test_f, test_l)
            #ensemble(train_f, train_l, test_f, test_l)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
