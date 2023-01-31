import claro_class as cl
import pandas as pd
import os
import fnmatch
import sys

"""
Author : Jacopo Altieri
Usage: from the command line, provide a path to a single claro file, a list of files or a directory.
If given a single Claro, prints the data, the linear interpolation and the erf-evaluated coefficients. Also plots these values.
If given a directory, creates a list of files matching the path (i.e. Summary\s_curve) and then proceed to analyze them.
If given a list of files, reads it and analyzes them, creating an output data sheet as well as distribution histograms.
"""

# function to check if the input is a single claro file
def isSingle(path):
        singlename = "*Ch_?_offset_?_Chip_???*"
        return fnmatch.fnmatch(path, singlename) #bool




# check if path has been given
if len(sys.argv) !=2:
    print("\nUsage: insert a valid directory or filename\n")
    sys.exit(1)

path = sys.argv[1]


# Apply class method based on the input file

if isSingle(path):
    print (f'Provided a single Claro file, analyzing...\n')
    single = cl.Single(path)
    single.fit_lin()
    single.fit_erf()    # default arguments: (fit_guess:dict=None)
    single.printData()
    single.plotter()    # default arguments: (scatter = True, show_lin = True, show_erf = True, saveplot = False)
    sys.exit(0)         # the program ends here if given a single file
    
    
elif os.path.isdir(path):
    print (f'Provided a directory, analyzing...\n')
    multi = cl.Claro(path)
    multi.dir_walker()
    
else:
    print(f'provided a list of directories, analyzing...\n')
    multi = cl.Claro(path)
    multi.list_reader()

# read and store all the useful data
multi.analyzer()    # default arguments: (discard_unfit = True)

# create the histograms
multi.histograms()  # default arguments: (saveplot = True)