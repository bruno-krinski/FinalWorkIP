#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sources.classes import *


def readImages(file_name):
    images_file = File(file_name, "r")
    images = []
    while True:
        l, tam = images_file.readList()
        if tam == 0:
            return images
        else:
            images.append(Image(l[0], l[1]))


def glcmFeaturesVector(images, images_path, output_file_name, method):
  output_file = File("features/" + output_file_name, "w+")
 
  glcm = GLCM([1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, method)
  img = cv2.imread(images_path + "/" + images[0].name, 0)
  hist = glcm.describe(img)
  print(hist[0])


def lbpFeaturesVector(images, images_path, output_file_name, method):
    output_file = File("features/" + output_file_name, "w+")

    lbp = LBP(1, method)
    
    for image in images:
        img = cv2.imread(images_path + "/" + image.name, 0)
        image.features = lbp.describe(img)
        #print(len(image.features))
        s = image.label + " "
        for f in image.features:
        	s += str(f) + " "
        s += "\n"
        output_file.writeString(s)


def extractFeatures():
    file_name = 'training.txt'
    images_path = '../TrainVal'

    images = readImages(file_name)
    """
    method = ['default', 'ror', 'uniform', 'nri_uniform']
    for m in method:
    	output_file_name = 'lbp_' + m + '_train.txt'  
    	lbpFeaturesVector(images, images_path, output_file_name, m)
    """
    method = ['contrast','dissimilarity','homogeneity',
              'energy', 'correlation', 'ASM']
    for m in method:
      output_file_name = 'glcm_' + m + '_train.txt'
      glcmFeaturesVector(images, images_path, output_file_name, m)
