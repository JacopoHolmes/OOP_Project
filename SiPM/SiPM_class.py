import pandas as pd
import fnmatch
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


###############################################################################
#                                Single file analyzer                         #
###############################################################################


class Single:
    # Constructor definition
    def __init__(self, path):
        self.path = path
        self._fileinfo = {}

    # fileinfo retriever method
    def _get_fileinfo(self):
        path = self.path

        _ardu = re.search(".+ARDU_(.+?)_.+", path).group(1)
        _direction = re.search(".+[0-9]_(.+?)_.+", path).group(1)
        _test = re.search(".+Test_(.+?)_.+", path).group(1)
        _temp = re.search(".+_(.+?)_dataframe.+", path).group(1)

        self._fileinfo = {
            "direction": _direction,
            "ardu": _ardu,
            "test": _test,
            "temp": _temp,
        }
        return self._fileinfo

    # Data reader method
    def reader(self):
        path = self.path
        self._get_fileinfo()

        # Skip rows until header (useful if the file contains comments on top)
        def header_finder(path):
            with open(path, "r") as file:
                for num, line in enumerate(file, 0):
                    if "SiPM" in line:
                        return num
                    else:
                        print("Error : header not found")
                        sys.exit(1)

        df = pd.read_csv(path, header=header_finder(path))
        self.df_sorted = df.sort_values(by=["SiPM", "Step"], ignore_index=True)

    # Analyzer and plotter method
    def analyzer(self, f_starting_point=1.55):
        start = f_starting_point

        # Forward analyzer
        if self._fileinfo["direction"] == "f":
            df_grouped = self.df_sorted.groupby("SiPM")
            results = df_grouped.apply(fwd_analyzer, start)
            joined_df = self.df_sorted.join(results, on="SiPM")

            # saving R_quenching to .csv for each SiPM
            out_df = joined_df[
                ["SiPM", "R_quenching", "R_quenching_std"]
            ].drop_duplicates(subset="SiPM")
            res_fname = rf"Arduino {self._fileinfo['ardu']} Test {self._fileinfo['test']} Forward results.csv"
            out_df.to_csv(res_fname, index=False)
            print(f"Results saved as {os.getcwd()}\{res_fname}")

            # Plotting
            pdf_name = f"Arduino {self._fileinfo['ardu']} Test {self._fileinfo['test']} Forward.pdf"
            pdf_fwd = PdfPages(pdf_name)
            joined_df.groupby("SiPM").apply(fwd_plotter, pdf_fwd)
            print(f"Plot saved as {os.getcwd()}\{pdf_name}")
            pdf_fwd.close()

        # Reverse analyzer
        else:
            pdf_name = f"Arduino {self._fileinfo['ardu']} Test {self._fileinfo['test']} Reverse.pdf"
            pdf_rev = PdfPages(pdf_name)

            df_grouped = self.df_sorted.groupby("SiPM")
            df_grouped.apply(rev_plotter, pdf_rev)
            pdf_name = f"Arduino {self._fileinfo['ardu']} Test {self._fileinfo['test']} Reverse.pdf"

            print(f"Plot saved as {os.getcwd()}\{pdf_name}")
            pdf_rev.close()
            pass


######################################################################
#           Mathematical functions and other static methods          #
######################################################################


@staticmethod
def fwd_analyzer(data, starting_point):
    """Linear regression"""

    x = data["V"]
    y = data["I"]
    # isolate the linear data
    x_lin = x[x >= starting_point]
    y_lin = y[x >= starting_point]

    # linear regression
    model = stats.linregress(x_lin, y_lin)
    m = model.slope
    q = model.intercept
    R_quenching = 1000 / m
    R_quenching_std = max(
        model.stderr, 0.03 * R_quenching
    )  # overestimation of the R standard dev

    # saving the values
    values = pd.Series(
        {
            "R_quenching": R_quenching,
            "R_quenching_std": R_quenching_std,
            "start": starting_point,
            "m": m,
            "q": q,
        }
    )
    return values


@staticmethod
def fwd_plotter(data, pdf):
    """Plot to pdf"""
    data["y_lin"] = (
        data["m"] * data["V"] + data["q"]
    )  # find y values via linear regression

    fig, ax = plt.subplots()
    sipm_number = list(data["SiPM"].drop_duplicates())[0]
    fig.suptitle(f"Forward IV curve: SiPM {sipm_number}")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current(mA)")
    ax.grid("on")

    ax.errorbar(data["V"], data["I"], data["I_err"], marker=".", zorder=1)
    ax.plot(
        data[data["V"] >= data["start"]]["V"],
        data[data["V"] >= data["start"]]["y_lin"],
        color="green",
        linewidth=1.2,
        zorder=2,
    )  # the conditions are there to plot only on the linear part of the curve
    ax.annotate(
        f'Linear fit: Rq = ({data["R_quenching"].iloc[0]:.2f} $\pm$ {data["R_quenching_std"].iloc[0]:.2f}) $\Omega$',  # iloc to take only one value
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        color="black",
    )

    pdf.savefig()
    plt.close()


@staticmethod
def rev_analyzer(data):
    pass


@staticmethod
def rev_plotter(data, pdf):
    fig, ax = plt.subplots()
    fig.suptitle("Reverse IV curve")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current(mA)")
    ax.grid("on")

    ax.set_yscale("log", nonpositive="clip")
    ax.errorbar(data["V"], data["I"], data["I_err"], marker=".")
    pdf.savefig()
    plt.close()


@staticmethod
def progress_bar(progress, total):
    """Provides a visual progress bar on the terminal"""

    percent = int(100 * (progress / float(total)))
    bar = "%" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar} | {percent:.2f}%", end="\r")


###############################################################################
#                                Directory analyzer                           #
###############################################################################


class dir_reader:
    # Constructor definition
    def __init__(self, dir):
        self.dir = dir

    # File finder method
    def dir_walker(self):
        top = self.path
        name_to_match = "ARDU_*_dataframe.csv"
        file_list = []

        for root, dirs, files in os.walk(top):
            for file in files:
                full_path = os.path.join(root, file)
                if fnmatch.fnmatch(full_path, name_to_match):
                    file_list.append(full_path)

        self.__file_list = file_list
        return self.__file_list
