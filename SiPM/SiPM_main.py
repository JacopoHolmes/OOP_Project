import SiPM_class as sipm
import sys
import os


"""
Author: Jacopo Altieri

This program analyzes SiPM data from either a single .csv file or a directory containing multiple .csv files.
For a single SiPM file, it reads the Arduino info, data and the direction, and then produces the following analysis for each SiPM in the arduino (30 total):
- If the direction is forward, it produces a linear regression on the linear part of the curve, an plots it.
  The slope of this line is the Quenching Resistance (rescaled by some constant factors).
- If the direction is reverse, it evaluates the derivative of the data, fits a 5th-degree polynomial on it and then a gaussian curve on top of the polynomial peak.
  The mean of this gaussian is the Breakdown Voltage of the SiPM. It also produces a plot with all these data.
If a directory path is provided, the program will traverse the directory and analyze all .csv files within it. The analyzed data is then saved to the "savepath" directory.
On top of the single analysis, histograms of all the R_q and V_bd are also generated, with the option to compare temperature and day differences.

Usage: 
----------
    $ python .\SiPM_main.py <input_file/input_directory>

Inputs:
----------
    input_file/input_directory: str
        Path to a single ARDU file or a directory containing ARDU files.

Outputs:
----------
    (single file analysis)
        .csv containing the evaluated data (R_q or V_bd based on the direction)
        .pdf file with all the SiPM plots
    (multiple file analysis)
        "results" folder, containing:
            subfolder for each dataset, containing .csv and .pdf files of each ARDU file
            Histograms of the evaluated data for each dataset and (if wanted) comparison between LN2 measurements and measurements made on the same day.

Dependencies:
----------
    SiPM_class.py
    sys
    os
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
    directory.dir_analyzer()  # Default arguments: (root_savepath = os.getcwd() ,hide_progress=True)
    directory.histograms()  # Default arguments (compare_temp=True , compare_day=True)


else:
    print("Provided a file path, analyzing...")
    single = sipm.Single(path)
    single.reader()
    single.analyzer()  # Default arguments: (room_f_start=0.75, ln2_f_start=1.55, peak_width=10, savepath=os.getcwd(), hide_progress=False)