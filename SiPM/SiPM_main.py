import SiPM_class as sipm
import pandas as pd
import sys
import os


"""
Author : Jacopo Altieri
Usage : Provide a .csv file or a directory to analyze.
"""

# Check if a valid argument is given
if len(sys.argv) != 2:
    print("\nUsage: insert a valid path to a .csv file or directory\n")
    sys.exit(1)

path = sys.argv[1]

if os.path.isdir(path):
    print("Provided a directory path, analyzing...")
    sipm.dir_reader(path)

else:
    print("Provided a file path, analyzing...")
    single = sipm.Single(path)
    single.reader()
    single.analyzer()  # default arguments : (f_starting_point = 1.6)
