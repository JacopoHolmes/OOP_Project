import claro_class as cl
import numpy as np
import pandas as pd
import os
import fnmatch
import sys
import matplotlib.pyplot as plt


# Leave hardcoded = true to test the program on a single file, just for now
hardcoded = True
path = r'C:\Users\jacop\Desktop\OOP\Ch_7_offset_0_Chip_004.txt'

if hardcoded != True:
    # check if path has been given
        if len(sys.argv) !=2:
            print("\nUsage: insert a valid directory or filename\n")
        sys.exit(1)

        #path = sys.argv[1]

        def isFile(path):
            filename = "Ch_?_offset_?_Chip_00?"
            return fnmatch.fnmatch(path, filename) # True


prova = cl.Single(path)
prova.fit_erf()
prova.printData()
prova.plotter()



