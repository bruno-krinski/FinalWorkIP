#!usr/bin/python

import sys
import cv2
import tqdm
import time
import datetime as dt
import numpy as np
import mahotas as mh
from sklearn import preprocessing
from skimage import feature
from mahotas.features import lbp
from skimage import feature
import numpy as np
from scipy.stats import itemfreq
from collections import Counter

class LBP:
    def __init__(self, num_points, radius, method):
        self.num_points = num_points
        self.radius = radius
        self.method = method

    def describe(self, image):
        #return lbp(image, self.radius, self.num_points)
        lbp = feature.local_binary_pattern(image,self.num_points,
                                           self.radius,method=self.method)
        hist = itemfreq(lbp.ravel())
        return hist[:,1]

class Image:
    def __init__(self, name, label):
        self.name = name
        self.label = label

def read_image_file(file_name):
    image_file = open(file_name, "r")

    images = []
    while True:
        my_string = image_file.readline()
        if my_string == "":
            return images
        else:
            my_string = my_string.split()
            image = Image(my_string[0], my_string[1])
            images.append(image)

def describe_images(path, images,out_file):
    features = []
    lbp = LBP(8, 1, "nri_uniform")
    out = open(out_file, "w+")
    for image in images:
        img = cv2.imread(path + image.name, 0)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        features.append(lbp.describe(img))
        #print(lbp.describe(img))
  
    for i in range(0,len(features)):
        my_string = images[i].label + ' '        
        for f in features[i]:
            my_string += str(f) + ' '
        my_string += '\n'
        out.write(my_string)
        #print(features[0])
   # out.close()

def main(argv):
    if len(argv) != 4:
        print("Input Error!")
        print("Use mode: python gen_features.py <path to images file> <path to images> <out file>")
        exit()

    images = read_image_file(argv[1])
    describe_images(argv[2],images,argv[3])

if __name__ == "__main__":
    init = dt.datetime.now()
    main(sys.argv)
    end = dt.datetime.now()
    print(end - init)

