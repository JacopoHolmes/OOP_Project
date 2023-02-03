import pandas as pd
import fnmatch
import os
import sys
import re
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats, signal, optimize


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
        self.df_grouped = self.df_sorted.groupby("SiPM")

    # Analyzer and plotter method
    def analyzer(self, f_starting_point=1.55, peak_width=20):
        start = f_starting_point
        width = peak_width

        # Forward analyzer
        if self._fileinfo["direction"] == "f":
            results = self.df_grouped.apply(fwd_analyzer, start)
            joined_df = self.df_sorted.join(results, on="SiPM")

            # saving R_quenching to .csv for each SiPM
            out_df = joined_df[
                ["SiPM", "R_quenching", "R_quenching_std"]
            ].drop_duplicates(subset="SiPM")
            res_fname = rf"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Forward_results.csv"
            out_df.to_csv(res_fname, index=False)
            print(f"Results saved as {os.getcwd()}\{res_fname}")

            # Plotting
            pdf_name = f"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Forward.pdf"
            pdf_fwd = PdfPages(pdf_name)
            joined_df.groupby("SiPM").apply(fwd_plotter, pdf_fwd)
            print(f"Plot saved as {os.getcwd()}\{pdf_name}")
            pdf_fwd.close()

        # Reverse analyzer
        else:
            results = self.df_grouped.apply(rev_analyzer, width)
            joined_df = self.df_sorted.join(results, on="SiPM")

        # saving V_bd to .csv for each SiPM
            out_df = joined_df[
                ["SiPM", "V_bd", "V_bd_std"]
            ].drop_duplicates(subset="SiPM")
            res_fname = rf"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Reverse_results.csv"
            out_df.to_csv(res_fname, index=False)
            print(f"Results saved as {os.getcwd()}\{res_fname}")
            
            # Plotting
            pdf_name = f"Arduino {self._fileinfo['ardu']} Test {self._fileinfo['test']} Reverse.pdf"
            pdf_rev = PdfPages(pdf_name)
            
            self.df_grouped.apply(rev_plotter, pdf_rev)
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

    x = data["V"].to_numpy()
    y = data["I"].to_numpy()
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

    # Returning the values
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
        color="darkgreen",
        linewidth=1.2,
        zorder=2,
    )  # The conditions are there to plot only on the linear part of the curve
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
def rev_analyzer(data, peak_width: int):
    # Accessing the data
    x = data["V"].to_numpy()
    y = data["I"].to_numpy()

    # Evaluation of the 1st derivative
    dy_dx = np.gradient(y) / np.gradient(x)
    derivative = 1 / y * dy_dx

    # 5th degree polynomial fit
    fifth_poly = Polynomial.fit(x, derivative, 5)
    coefs = fifth_poly.coef

    # Peak finder
    peaks = signal.find_peaks(fifth_poly(x), width=peak_width)[
        0
    ]  # width parameter to discard smaller peaks
    idx_max = peaks[np.argmax(fifth_poly(x)[peaks])]
    x_max = x[idx_max]

    # Gaussian fit around the peak
    x_gauss = x[idx_max - int(peak_width / 2) : idx_max + int(peak_width / 2) + 1]
    y_gauss = fifth_poly(x)[
        idx_max - int(peak_width / 2) : idx_max + int(peak_width / 2) + 1
    ]
    std_estimate = x_gauss.ptp() / 2
    fit_guess = [0, 1, x_max, std_estimate]
    params, covar = optimize.curve_fit(gauss, x_gauss, y_gauss, fit_guess, maxfev=10000)
    mu = params[2]
    std = params[3]

    # Returning the values
    values = pd.Series(
        {
            "V_bd": mu,
            "V_bd_std": std,
            "width": std_estimate*2,
            "coefs": coefs,
        }
    )
    return values


@staticmethod
def rev_plotter(data, pdf):
    fig, ax = plt.subplots()
    sipm_number = list(data["SiPM"].drop_duplicates())[0]
    fig.suptitle(f"Reverse IV curve: SiPM {sipm_number}")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current(mA)")
    ax.grid("on")

    ax.set_yscale("log", nonpositive="clip")
    ax.errorbar(data["V"], data["I"], data["I_err"], marker=".")

    # ax.set_yscale("linear")
    # ax.plot(x, derivative , color="darkgreen", marker=".")
    pdf.savefig()
    plt.close()
    # plt.plot(x, derivative)
    # plt.plot(x_gauss,y_gauss , color ="black" , marker ="." , zorder = 10)
    # plt.plot(x_gauss , gauss(x_gauss, *params))
    #dy_dx = np.gradient(y) / np.gradient(x)
    #derivative = 1 / y * dy_dx
    # plt.show()


@staticmethod
def progress_bar(progress, total):
    """Provides a visual progress bar on the terminal"""

    percent = int(100 * (progress / float(total)))
    bar = "%" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar} | {percent:.2f}%", end="\r")


# Gaussian function
def gauss(x, H, A, mu, sigma):
    # Returns a gaussian curve with displacement H, amplitude A, mean mu and std sigma.
    return H + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


###############################################################################
#                                Directory analyzer                           #
###############################################################################


class DirReader:
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
