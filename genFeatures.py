#!usr/bin/python

import sys
import numpy as np
from skimage import feature
from libStrings import *

class LBP:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, mtd):
        lbp = feature.local_binary_pattern(image,
                                           self.numPoints,
                                           self.radius,
                                           method=mtd)
        return lbp

def main(argv):
    if(len(arg) != 1):
        print(inputError)
        print(gf_helpInput)
        exit()

    lbp = LBP(8,1)
    


if __name__ == "__main__":
    main(sys.argv)