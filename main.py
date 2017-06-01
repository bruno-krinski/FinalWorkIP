#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from classify import classify
from sources.classes import *
from sources.functions import *
from genFeatures import extractFeatures


def main(argv):
    if len(argv) != 1:
        print("Wrong number of parameters!")
        exit(0)

    print("Choose the mode:")
    mode = "0 - Validation mode.\n"
    mode += "1 - Test mode.\n"
    m = int(input(mode))
    if m == 0:
        options = "0 - Generate training file.\n"
        options += "1 - Extract features.\n"
        options += "2 - Classify.\n"
        options += "3 - Test All\n"
        ch = int(input(options))
        if ch == 0:
            generateResourceFile(7, 40, "training.txt")
        elif ch == 1:
            extractFeatures()
        elif ch == 2:
            classify()


if __name__ == "__main__":
    main(sys.argv)
