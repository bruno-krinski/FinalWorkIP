#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sources.classes import *


def writeFeatures(label, features, output_file):
    s = label + " "
    for f in features:
        s += str(f) + " "
    s += "\n"
    output_file.writeString(s)


def readImages(file_name):
    images_file = File(file_name, "r")
    images = []
    while True:
        l, tam = images_file.readList()
        if tam == 0:
            return images
        else:
            images.append(Image(l[0], l[1]))


def glcmFeaturesVector(images, images_path, output_file_name, dists):

    output_file = File("features/" + output_file_name, "w+")

    method = ['contrast','dissimilarity','homogeneity','energy','correlation',
              'ASM']

    for image in images:
        img = cv2.imread(images_path + "/" + image.name, 0)
        tot = []
        for m in method:
            glcm = GLCM(dists, [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, m)
            image.features = glcm.describe(img)
            aux = []
            for f in image.features:
                aux = np.concatenate((aux,f),axis=0)
            tot = np.concatenate((tot,aux),axis=0)

        writeFeatures(image.label, tot, output_file)


def lbpFeaturesVector(images, images_path):

    method = ['default', 'ror', 'uniform', 'nri_uniform']

    for m in method:
        lbp = LBP(1, m)
        output_file_name = 'lbp_' + m + '_train.txt'
        output_file = File("features/" + output_file_name, "w+")
        for image in images:
            img = cv2.imread(images_path + "/" + image.name, 0)
            image.features = lbp.describe(img)
            writeFeatures(image.label, image.features, output_file)


def haralickFeatures(images, images_path):
    h = Haralick()
    output_file = File("features/haralick_train.txt", "w+")
    for image in images:
        img = cv2.imread(images_path + "/" + image.name, 0)
        image.features = h.describe(img)
        #print(image.features)
        aux = []
        for f in image.features:
            aux = np.concatenate((aux,f),axis=0)
        writeFeatures(image.label, aux, output_file)

def extractFeatures():
    file_name = 'training.txt'
    images_path = '../Trees'

    images = readImages(file_name)

    haralickFeatures(images, images_path)

    lbpFeaturesVector(images, images_path)

    output_file_name = 'glcm_1_train.txt'
    glcmFeaturesVector(images, images_path, output_file_name,[1])
    output_file_name = 'glcm_2_train.txt'
    glcmFeaturesVector(images, images_path, output_file_name,[1,2])
    output_file_name = 'glcm_3_train.txt'
    glcmFeaturesVector(images, images_path, output_file_name,[1,2,3])
    output_file_name = 'glcm_4_train.txt'
    glcmFeaturesVector(images, images_path, output_file_name,[1,2,3,4])
    output_file_name = 'glcm_5_train.txt'
    glcmFeaturesVector(images, images_path, output_file_name,[1,2,3,4,5])
