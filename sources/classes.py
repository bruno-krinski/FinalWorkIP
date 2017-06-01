#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from skimage import feature
from scipy.stats import itemfreq
from skimage.feature import greycomatrix, greycoprops


class LBP:
  def __init__(self, radius, method):
    self.radius = radius
    self.num_points = 8 * radius
    self.method = method

  def describe(self, image):
    x = feature.local_binary_pattern(image, 
                                     self.num_points, 
                                     self.radius, 
                                     method=self.method)
    #histogram = itemfreq(x.ravel())
    #return histogram[:, 1]/sum(histogram[:, 1]	)
    #print(x)
    n_bins = int(x.max() + 1)
    hist,_ = np.histogram(x,normed=True,bins = n_bins,range=(0,n_bins))
    return hist

class GLCM:
  def __init__(self, distances, angles, levels, method):
    self.distances = distances
    self.angles = angles
    self.method = method
    self.levels = levels
 
  def describe(self, image):
    x = greycomatrix(image, 
                     self.distances, 
                     self.angles, 
                     levels=self.levels,
                     normed=True, 
                     symmetric=True)
    x = greycoprops(x, self.method)
    return x
                

class Image:
  def __init__(self, name, label):
    self.name = name
    self.label = label
    self.features = []


class File:
  def __init__(self, name, mode):
    self.__file = open(name, mode)

  def __del__(self):
    self.__file.close() 

  def writeString(self, s):
    self.__file.write(s)

  def readList(self):
    aux = self.__file.readline().split()
    return aux, len(aux)

  def writeList(self, list_to_write):
    for l in list_to_write:
      self.__file.write(l)
