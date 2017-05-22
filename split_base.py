#!usr/bin/python

"""This module is responsible to split the base of trees in 60% and 40%"""

import random

def main():
    """Funcion main"""
    labels = []
    labels.append("Apuleia")
    labels.append("Aspidosperma")
    labels.append("Astronium")
    labels.append("Byrsonima")
    labels.append("Calophyllum")
    labels.append("Cecropia")
    labels.append("Cedrelinga")
    labels.append("Cochlospermum")
    labels.append("Combretum")
    labels.append("Copaifera")

    train = open("training.txt", "w+")
    val = open("validation.txt", "w+")

    for label in labels:
        for i in range(0, 7):
            for j in range(0, 40):
                if j < 10:
                    image_name = label + "_00" + str(i) + "_00" + str(j) + ".tif"
                else:
                    image_name = label + "_00" + str(i) + "_0" + str(j) + ".tif"
                rand = random.randint(1, 10)
                if rand <= 6:
                    train.write(image_name + " " + label + "\n")
                else:
                    val.write(image_name + " " + label + "\n")
    train.close()
    val.close()

if __name__ == "__main__":
    main()
