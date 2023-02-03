import os
import re
import fnmatch
import pandas as pd
import numpy as np
from scipy import optimize, special, stats
import matplotlib.pyplot as plt
import warnings


"""Single and multiple file analyzer share some static methods, so they must go together"""


###############################################################################
#                                Single file analyzer                         #
###############################################################################
class Single:
    # Constructor definition
    def __init__(self, path):
        self.path = path
        data = _get_data(path)
        self.height = data["height"]
        self.t_point = data["t_point"]
        self.width = data["width"]
        self.x = data["x"]
        self.y = data["y"]
        self._metadata = data["meta"]
        self.fit_guess = data["fit_guess"]

    # linear fit method
    def fit_lin(self):
        # vector parsing (allows for base and saturation values to be different than 0 and 1000)
        x_int = self.x
        y_int = self.y
        while y_int[0] == y_int[1]:
            x_int = np.delete(x_int, 0)
            y_int = np.delete(y_int, 0)
        while y_int[-1] == y_int[-2]:
            x_int = np.delete(x_int, -1)
            y_int = np.delete(y_int, -1)
        self.x_int = x_int
        self.y_int = y_int

        # linear regression and t-point eval
        model = stats.linregress(x_int, y_int)

        half_maximum = (y_int[-1] - y_int[0]) / 2
        trans_lin = (half_maximum - model.intercept) / model.slope
        self.half_max = half_maximum
        self.trans_lin = trans_lin

        # saving the values
        values = {
            "slope": model.slope,
            "intercept": model.intercept,
            "transition point (Linear)": trans_lin,
            "R_squared": model.rvalue**2,
        }
        return values

    # erf fit method
    def fit_erf(self, fit_guess: dict = None):
        function = modified_erf
        x = self.x
        y = self.y
        if fit_guess is None:
            fit_guess = self.fit_guess
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Covariance of the parameters could not be estimated"
            )
            params, covar = optimize.curve_fit(function, x, y, fit_guess, maxfev=10000)

            erf_fit_x = np.linspace(self.x.min(), self.x.max(), 100)
            erf_fit_y = function(erf_fit_x, *params)
            self.erf_x = erf_fit_x
            self.erf_y = erf_fit_y
            std = np.sqrt(np.diag(covar))

        if np.isinf(std[1]) or np.isnan(std[1]):
            std[0] = np.nan
            std[1] = np.nan
            std[2] = np.nan
        self.erf_params = {
            "height": [params[0], std[0]],
            "transition point (erf)": [params[1], std[1]],
            "width": [params[2], std[2]],
        }
        return self.erf_params

    # data printing method
    def printData(self):
        for key, value in self._metadata.items():
            print(key, ": ", value)
        print("\n")
        for key, value in self.fit_lin().items():
            print(key, ": ", value)
        for key, value in self.fit_erf().items():
            print(key, ": ", value)
        if self.fit_erf()["transition point (erf)"][1] == 0:
            print("the fit did not converge, std set to 0.")
        print("\n")

    # plotter method
    def plotter(self, scatter=True, show_lin=True, show_erf=True, saveplot=False):
        # retrieving the data to be plotted
        fileinfo = _get_fileinfo(self.path)
        meta = self._metadata
        x = self.x
        y = self.y
        half_max = self.half_max
        x_int = self.x_int
        lin_intercept = self.fit_lin().get("intercept")
        lin_slope = self.fit_lin().get("slope")
        lin_y = lin_slope * x_int + lin_intercept
        t_lin = self.trans_lin
        erf_x = self.erf_x
        erf_y = self.erf_y
        t_erf = self.erf_params["transition point (erf)"][0]
        t_erf_std = self.erf_params["transition point (erf)"][1]
        w_erf = self.erf_params["width"][0]
        w_erf_std = self.erf_params["width"][1]

        # plot options
        fig, ax = plt.subplots()
        fig.suptitle(
            f"Fit Claro: Station {fileinfo['station']}, Chip {fileinfo['chip']}, Channel {fileinfo['channel']}"
        )
        ax.set_xlabel("ADC")
        ax.set_ylabel("Counts")
        ax.grid("on")

        # plotting based on arguments (default all True)
        if scatter == True:
            ax.scatter(x, y, color="black", marker=".")
            ax.annotate(
                f"From data file:\nTransition point = {meta['transition point']:.2f}\n"
                + f"Width = {meta['width']:.2f}",
                xy=(0.025, 0.975),
                xycoords="axes fraction",
                verticalalignment="top",
                color="black",
                alpha=0.8,
            )

        if show_lin == True:
            ax.plot(x_int, lin_y, color="green")
            ax.scatter(t_lin, half_max, color="green", marker="o")
            ax.annotate(
                f"From linear interp.:\nTransition point = {t_lin:.2f}\n",
                xy=(0.025, 0.750),
                xycoords="axes fraction",
                verticalalignment="top",
                color="green",
                alpha=0.8,
            )

        if show_erf == True:
            ax.plot(erf_x, erf_y, color="blue")
            ax.scatter(t_erf, half_max, color="blue", marker="s")
            ax.annotate(
                f"From erf fit:\nTransition point = {t_erf:.2f} $\pm$ {t_erf_std:.2f}\n"
                + f"Width = {w_erf:.2f} $\pm$ {w_erf_std:.2f}\n",
                xy=(0.025, 0.6),
                xycoords="axes fraction",
                verticalalignment="top",
                color="blue",
                alpha=0.8,
            )

        # plot saving (default False)
        if saveplot == True:
            plotname = f"Plot_Claro_Chip{fileinfo['chip']}_Ch{fileinfo['channel']}.png"
            plt.savefig(plotname, bbox_inches="tight")
            print(f"Plot saved as {os.getcwd()}\{plotname}")

        plt.show()


##############################################################################################################################################################


###############################################################################
#                                Directory analyzer                           #
###############################################################################


class Claro:
    # Constructor definition
    def __init__(self, path):
        self.path = path
        self.__file_list = []

    # Directory walker method
    def dir_walker(self):
        top = self.path
        name_to_match = "*Station*_Summary/Chip_*/S_curve/Ch_*_offset_*_Chip_*.txt"
        file_list = []

        for root, dirs, files in os.walk(top):
            for file in files:
                full_path = os.path.join(root, file)
                if fnmatch.fnmatch(full_path, name_to_match):
                    file_list.append(full_path)

        with open(r".\claro_allfiles.txt", "w") as outfile:
            outfile.write("\n".join(file_list))
        print(f"found {len(file_list)} files to read...")
        print(rf"list of files to analyze created as {os.getcwd()}\claro_allfiles.txt")
        self.__file_list = file_list
        return self.__file_list

    # Directory list reader method
    def list_reader(self):
        list = self.path
        with open(list, "r") as all_files:
            file_list = all_files.readlines()
        self.__file_list = file_list
        return self.__file_list

    # List analyzer method
    def analyzer(self, discard_unfit=True):
        """Reads file info and data and store them into files.
        If discard_unfit = True  separates all the files that didn't converge; otherwise put their tr. point std to 0
        """

        if discard_unfit == True:
            with open(r".\claro_unfit_chips.txt", "w") as create_unfit:
                pass

        list = self.__file_list
        _goodfiles = []
        _badfiles = []
        _t_list = []
        _erf_t_list = []

        # split good and bad files and write them to files
        for idx, element in enumerate(list):
            chip_name = element.strip("\n")
            with open(chip_name, "r") as chip:
                if re.search("[a-zA-Z]", chip.readline()):
                    _badfiles.append(chip_name)
                    continue
                _goodfiles.append(chip_name)
            progress_bar(idx + 1, len(list))

        print("\n")
        print(f"found {len(_badfiles)} bad files")
        print(rf"list of bad files created as {os.getcwd()}\claro_badfiles.txt")
        with open(r".\claro_badfiles.txt", "w") as outfile:
            outfile.write("\n".join(_badfiles))

        print(f"found {len(_goodfiles)} good files")
        print(rf"list of good files created as {os.getcwd()}\claro_goodfiles.txt")
        with open(r".\claro_goodfiles.txt", "w") as outfile:
            outfile.write("\n".join(_goodfiles))

        # read data and evaluate erf t.point from good files
        print("processing the good files...")

        with open(r".\claro_processed_chips.txt", "w") as processed:
            processed.write(
                "Station\tChip\tChannel\tAmplitude\tTr. Point\tWidth\terf_tr.point\tStd_erf_tr.point\n"
            )

        for idx, file in enumerate(_goodfiles):
            chip_name = file.strip("\n")
            with open(chip_name, "r") as chip:
                # get data
                data = _get_data(chip_name)
                info = _get_fileinfo(chip_name)
                all_data = info | data["meta"]

                # evaluate erf t. point
                def fit_erf(self):
                    function = modified_erf
                    x = data["x"]
                    y = data["y"]

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Covariance of the parameters could not be estimated",
                        )
                        warnings.filterwarnings(
                            "ignore", message="invalid value encountered in sqrt"
                        )

                        params, covar = optimize.curve_fit(
                            function, x, y, data["fit_guess"], maxfev=1000
                        )
                        erf_fit_x = np.linspace(x.min(), x.max(), 100)
                        erf_fit_y = function(erf_fit_x, *params)
                        std = np.sqrt(np.diag(covar))

                    if np.isinf(std[1]) or np.isnan(std[1]):
                        std[1] = 0

                    t_erf = [params[1], std[1]]
                    return t_erf

                t_erf = fit_erf(self)
                all_data["t_erf"] = t_erf[0]
                all_data["std_t_erf"] = t_erf[1]

                _t_list.append(all_data["transition point"])
                _erf_t_list.append(all_data["t_erf"])
                self._t_list = _t_list
                self._erf_t_list = _erf_t_list
                # write data to file
                if discard_unfit == True:
                    if t_erf[1] == 0:
                        with open(r".\claro_unfit_chips.txt", "a") as unfit:
                            unfit.write("{}\n".format(all_data["path"]))
                            continue

                with open(r".\claro_processed_chips.txt", "a") as processed:
                    del all_data["path"]
                    values = all_data.values()
                    [
                        processed.write("{}\t".format(value))
                        for value in all_data.values()
                    ]
                    processed.write("\n")
                    progress_bar(idx + 1, len(_goodfiles))

        print("\n")
        print(rf"results stored in {os.getcwd()}\claro_processed_chips.txt")
        if discard_unfit == True:
            print(
                rf"list of unfit files created as {os.getcwd()}\claro_unfit_chips.txt"
            )

    # histogram maker method
    def histograms(self, saveplot=True):
        t = np.array(self._t_list)
        erf_list = np.array(self._erf_t_list)
        discrepancy = t - erf_list

        # plot options
        fig, axs = plt.subplots(3)
        fig.suptitle("Histogram of the transition points distribution")
        [ax.grid("on") for ax in axs]

        axs[0].hist(t, bins=200, color="green")
        axs[1].hist(erf_list, bins=200, color="blue")
        axs[2].hist(discrepancy, bins=10, range=(-1e-6, 1e-6))  ## Cambia bins a 10

        axs[0].set_title("Read T. point")
        axs[1].set_title("Erf T. point")
        axs[2].set_title("Discrepancy")
        plt.subplots_adjust(
            left=0.133, bottom=0.11, right=0.91, top=0.88, wspace=0.08, hspace=0.558
        )

        if saveplot == True:
            plotname = f"Histogram_transition_points.png"
            plt.savefig(plotname, bbox_inches="tight")
            print(f"Plot saved as {os.getcwd()}\{plotname}")

        plt.show()

    ######################################################################
    #           Mathematical functions and other static methods          #
    ######################################################################


def modified_erf(x, height, a, b):
    # Returns a modified erf function, shifted on the vertical axis by height/2, with parameters a and b.
    return (height / 2) * (1 + special.erf((x - a) / (b / 2 * np.sqrt(2))))


@staticmethod
def _get_fileinfo(path):
    """Retrieves Station chip and channel number from the path"""

    try:
        _chip = re.search(".+Chip_(.+?).txt", path).group(1)
        _channel = re.search(".+Ch_(.+?)_.+", path).group(1)
        _station = re.search(".+\Station_1__(.+?)_Summary.+", path).group(1)
    except AttributeError:
        _station = "?"

    _fileinfo = {"station": _station, "chip": _chip, "channel": _channel}
    return _fileinfo


@staticmethod
def _get_data(path):
    data = pd.read_csv(path, sep="\t", header=None, skiprows=None)
    height = data[0][0]
    t_point = data[1][0]
    width = np.abs(data[2][0])
    x = data.iloc[2:, 0].to_numpy()
    y = data.iloc[2:, 1].to_numpy()
    _metadata = {
        "path": path,
        "amplitude": height,
        "transition point": t_point,
        "width": width,
    }
    fit_guess = [height, t_point, width]

    all_data = {
        "height": height,
        "t_point": t_point,
        "width": width,
        "x": x,
        "y": y,
        "meta": _metadata,
        "fit_guess": fit_guess,
    }
    return all_data


@staticmethod
def progress_bar(progress, total):
    """Provides a visual progress bar on the terminal"""

    percent = int(100 * (progress / float(total)))
    bar = "%" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar} | {percent:.2f}%", end="\r")
