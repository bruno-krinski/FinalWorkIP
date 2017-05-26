import os

for i in range(0,10):
    os.system("python split_base.py")
    os.system("python gen_features.py training.txt ..\TrainVal\ features_train.txt")
    os.system("python gen_features.py validation.txt ..\TrainVal\ features_val.txt")
    #os.system("python classify.py knn features_train.txt features_val.txt")
    os.system("python classify.py svm features_train.txt features_val.txt")