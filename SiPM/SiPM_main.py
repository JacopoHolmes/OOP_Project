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
    directory = sipm.DirReader(path)
    directory.dir_walker()
    directory.dir_analyzer() # Default arguments: (root_savepath = os.getcwd() ,hide_progress=True)
    directory.histograms()  # Default arguments (compare_temp=True , compare_day=True)
    

else:
    print("Provided a file path, analyzing...")
    single = sipm.Single(path)
    single.reader()
    single.analyzer()  # Default arguments: (room_f_start=0.75, ln2_f_start=1.55, peak_width=10, savepath=os.getcwd(), hide_progress=False)