#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
from sources.classes import File


labels = ['Apuleia',
          'Aspidosperma',
          'Astronium',
          'Byrsonima',
          'Calophyllum',
          'Cecropia',
          'Cedrelinga',
          'Cochlospermum',
          'Combretum',
          'Copaifera']



def generateResourceFile(n, m, name):
    resource = []
    for label in labels:
        for i in range(0, n):
            for j in range(0, m):
                if j < 10:
                    image_name = label + "_00" + str(i) + "_00" + str(j) + ".tif"
                else:
                    image_name = label + "_00" + str(i) + "_0" + str(j) + ".tif"
                resource.append(image_name + " " + label + "\n")
    random.shuffle(resource)
    resource_file = File(name, "w+")
    resource_file.writeList(resource)


