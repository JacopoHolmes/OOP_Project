"""
Author : Jacopo Altieri

This program analyzes single Claro files or multiple Claro files in a directory.
For a single Claro file, it fits the data with linear and error function models,
prints the relevant information, and generates a plot of the data with the fits.
For multiple Claro files in a directory, it analyzes the data for all files,
generates histograms of the fit parameters, and stores the information in a summary dataframe.

Usage: 
----------
    $ python .\claro_main.py <input_file/input_directory>

Inputs:
----------
    input_file/input_directory: str
        Path to a single Claro file or a directory containing Claro files.

Outputs:
----------
    (single file analysis)
        Fit information, printed to the terminal.
        Plot of the data with the fits, saved in the working directory.
    (multiple file analysis)
        Summary dataframe, stored in the working directory.
        Histograms of the fit parameters, saved in the working directory.

Dependencies:
----------
    claro_class.py
    os
    fnmatch
    sys
"""


import claro_class as cl
import os
import fnmatch
import sys


def isSingle(path):
    """Checks if the path given is of a Single Claro file

    Args:
    ----------
        path (str): File path

    Returns:
    ----------
        bool: True if the path is of a single Claro
    """
    singlename = "*Ch_?_offset_?_Chip_???*"
    return fnmatch.fnmatch(path, singlename)


# check if path has been given
if len(sys.argv) != 2:
    print("\nUsage: insert a valid directory or filename\n")
    sys.exit(1)

path = sys.argv[1]


# Apply class method based on the input file
if isSingle(path):
    print(f"Provided a single Claro file, analyzing...\n")
    single = cl.Claro(path)
    single.fit_erf()    # default arguments: (fit_guess = None)
    single.print_data() 
    single.plotter()  # default arguments: (scatter=True, show_lin=True, show_erf=True, saveplot=False)
    sys.exit(0)  # the program ends here if given a single file

elif os.path.isdir(path):
    print(f"Provided a directory, analyzing...\n")
    multi = cl.MultiAnalyzer(path)
    multi.dir_walker_texas_ranger()

else:
    print(f"provided a list of directories, analyzing...\n")
    multi = cl.MultiAnalyzer(path)
    multi.list_reader()

multi.analyzer()  # default arguments: (discard_unfit=True, savepath=os.getcwd() ,erf_guess=None)
multi.histograms()  # default arguments: (saveplot=True)
