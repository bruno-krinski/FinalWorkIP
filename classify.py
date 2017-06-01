#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
from sklearn import svm
from sources.classes import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
        if tam == 0:
            return data_features, data_labels
        else:
            data_labels.append(x[0])
            data_features.append(x[1:len(x)])


def knn_function(train_features, train_labels, test_features, test_labels):
  knn = knc(n_neighbors=5)
  knn.fit(train_features, train_labels)
  result = knn.predict(test_features)
  print("KNN")
  print("Confusion Matrix: ")
  print(confusion_matrix(test_labels,result))
  print("Accuracy: ",accuracy_score(test_labels,result))


def svm_function(train_features, train_labels, test_features, test_labels):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    #svr = svm.SVC(C=10, kernel='linear')
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters).fit(train_features, train_labels)
    # print(clf.best_params_)
    result = clf.predict(test_features)
    print("SVM")
    print("Confusion Matrix: ")
    print(confusion_matrix(test_labels,result))
    print("Accuracy: ",accuracy_score(test_labels,result))

def mlp_function(train_features, train_labels, test_features, test_labels):
  mlp = MLPClassifier()
  mlp.fit(train_features, train_labels)
  result = mlp.predict(test_features)
  print("MLP")
  print("Confusion Matrix: ")
  print(confusion_matrix(test_labels,result))
  print("Accuracy: ",accuracy_score(test_labels,result))


def classify():

  """data = ['features/lbp_default_train.txt',
          'features/lbp_ror_train.txt',
          'features/lbp_uniform_train.txt',
          'features/lbp_nri_uniform_train.txt',
          'features/glcm_train.txt']"""
  data = ['features/lbp_default_train.txt']
	
  for d in data:
    print('======================================================================')
    print(d)
    data_features, data_labels = readData(d)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_features = min_max_scaler.fit_transform(data_features)

    for i in range(0, 1):
      rand = random.randint(1, 100)
      train_features,test_features,train_labels,test_labels = train_test_split(
                                                              data_features,
                                                              data_labels,
                                                              test_size=0.4,
                                                              random_state=rand)
      
      mlp_function(train_features, train_labels, test_features, test_labels)
      #knn_function(train_features, train_labels, test_features, test_labels)
    print('======================================================================')	        
