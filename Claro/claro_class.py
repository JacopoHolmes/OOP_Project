import os
import re
import fnmatch
import pandas as pd
import numpy as np
from scipy import optimize, special, stats
import matplotlib.pyplot as plt
import warnings


###############################################################################
#                                Single file analyzer                         #
###############################################################################
class Claro:
    """
    A class for representing and analyzing Claro data.

    Parameters:
    ----------
    path (str): The file path of the Claro data file.

    Attributes:
    ----------
        height (float): The height of the data.
        t_point (float): The transition point of the data.
        width (float): The width of the data.
        x (numpy.ndarray): The x values of the data.
        y (numpy.ndarray): The y values of the data.
        fit_guess (list): The initial guess for the erf fit.
        _fileinfo (dict): The file information, including the station, chip, and channel.

    Methods:
    ---------
        get_fileinfo(): Extracts and returns file information from the file path.
        get_data(): Reads the data from the `self.path` file, extracts the necessary information and returns it as a dictionary.
        fit_lin(): Fits a linear regression to the data in the transition zone and returns its parameters as a dictionary.
        fit_erf(fit_guess=None): Fits a shifted and traslated erf function to the data and returns its parameters as a dictionary.
        print_data() : Prints the extracted and estimated data on terminal.
        plotter(): Plots the ADC vs Counts data.
    """

    def __init__(self, path):
        """
        Initialize the Claro object with the file path.

        Args:
        ----------
            path (str): The file path of the Claro data file.
        """

        self.path = path
        data = self.get_data()
        self.height = data["height"]
        self.t_point = data["t_point"]
        self.width = data["width"]
        self.x = data["x"]
        self.y = data["y"]
        self.fit_guess = data["fit_guess"]
        self._fileinfo = self.get_fileinfo()

    def get_fileinfo(self):
        """
        Extracts and returns file information from the file path.

        Returns:
        ----------
            fielinfo (dict): A dictionary with keys 'station', 'chip', and 'channel' containing the extracted information.
        """
        try:
            _chip = re.search(".+Chip_(.+?).txt", self.path).group(1)
            _channel = re.search(".+Ch_(.+?)_.+", self.path).group(1)
            _station = re.search(".+\Station_1__(.+?)_Summary.+", self.path).group(1)
        except AttributeError:  # Station missing if given a single file to read
            _station = "?"

        return {"station": _station, "chip": _chip, "channel": _channel}

    def get_data(self):
        """
        This method reads the data from the `self.path` file, extracts the necessary information and returns it as a dictionary.

        Returns:
        ----------
            all_data (dict): A dictionary containing the following information:
                path (str): The file path of the Claro data file.
                height (float): The height of the data.
                t_point (float): The transition point of the data.
                width (float): The width of the data.
                x (np.array): The x-axis values of the data.
                y (np.array): The y-axis values of the data.
                fit_guess (list): A list containing the following information:
                    height (float): The height of the data.
                    t_point (float): The transition point of the data.
                    width (float): The width of the data.
        """
        data = pd.read_csv(self.path, sep="\t", header=None, skiprows=None)
        height = data[0][0]
        t_point = data[1][0]
        width = np.abs(data[2][0])
        x = data.iloc[2:, 0].to_numpy()
        y = data.iloc[2:, 1].to_numpy()
        fit_guess = [height, t_point, width]

        self.all_data = {
            "path": self.path,
            "height": height,
            "t_point": t_point,
            "width": width,
            "x": x,
            "y": y,
            "fit_guess": fit_guess,
        }
        return self.all_data

    def fit_lin(self):
        """
        Fits a linear regression to the data stored in self.x and self.y.

        Returns:
        ----------
            values (dict): A dictionary containing the following information:
                slope (float)
                intercept (float)
                transition_point_(Linear) (float)
                R_squared (float)"""
        # Vector parsing (allows for base and saturation values to be different than 0 and 1000)
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

        model = stats.linregress(x_int, y_int)
        self.half_max = (y_int[-1] - y_int[0]) / 2
        self.trans_lin = (self.half_max - model.intercept) / model.slope

        values = {
            "slope": model.slope,
            "intercept": model.intercept,
            "transition_point_(Linear)": self.trans_lin,
            "R_squared": model.rvalue**2,
        }
        return values

    def fit_erf(self, fit_guess=None):
        """
        Fits a shifted and traslated erf function to the data

        Args:
        ----------
            fit_guess (list, optional): a list containing the first guesses for the height, t_point and width of the data. Defaults to None.

        Returns:
        ----------
            self.erf_params (dict): A dictionary containing the following information:
                height (list): estimated height and its standard deviation
                transition_point_(erf): estimated transition point and its standard deviation
                width: estimated width and its standard deviation
        """

        if fit_guess is None:
            fit_guess = self.fit_guess

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Covariance of the parameters could not be estimated"
            )
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in sqrt"
            )
            params, covar = optimize.curve_fit(
                modified_erf, self.x, self.y, fit_guess, maxfev=10000
            )

            std = np.sqrt(np.diag(covar))

        if np.isinf(std[1]) or np.isnan(std[1]):
            std[0] = np.nan
            std[1] = np.nan
            std[2] = np.nan
        self.erf_params = {
            "height": [params[0], std[0]],
            "transition_point_(erf)": [params[1], std[1]],
            "width": [params[2], std[2]],
        }
        return self.erf_params

    def print_data(self):
        """
        Prints the height, transition point, and width of the data, as well as the fit results from `self.fit_lin()` and `self.fit_erf()`.
        Also prints a message if the fit for `self.fit_erf()` did not converge.

        Returns:
        ----------
            None
        """
        print(f"Height: {self.all_data['height']}")
        print(f"Transition point (file): {self.all_data['t_point']}")
        print(f"Width: {self.all_data['width']}")
        for key, value in self.fit_lin().items():
            print(key, ": ", value)
        for key, value in self.fit_erf().items():
            print(key, ": ", value)
        if self.fit_erf()["transition_point_(erf)"][1] == np.nan:
            print("the fit did not converge, std set to nan.")
        print("\n")

    def plotter(self, scatter=True, show_lin=True, show_erf=True, saveplot=False):
        """
        Plots the ADC vs Counts data along with the linear interpolation of the data,he fit of the data to an error function and annotates the transition point and width of the fit.

        Args:
        ----------
            scatter (bool, optional): If True, plot the original data as a scatter plot. Default is True.
            show_lin  (bool, optional): If True, plot the linear interpolation of the data. Default is True.
            show_erf (bool, optional): If True, plot the fit of the data to an error function. Default is True.
            saveplot (bool, optional): If True, save the plot as a PNG image in the current working directory. Default is False.

        Returns:
        ----------
            None
        """
        lin_intercept = self.fit_lin().get("intercept")
        lin_slope = self.fit_lin().get("slope")
        lin_y = lin_slope * self.x_int + lin_intercept
        h = self.erf_params["height"][0]
        t_erf = self.erf_params["transition_point_(erf)"][0]
        t_erf_std = self.erf_params["transition_point_(erf)"][1]
        w_erf = self.erf_params["width"][0]
        w_erf_std = self.erf_params["width"][1]

        # Plot options
        fig, ax = plt.subplots()
        fig.suptitle(
            f"Fit Claro: Station {self._fileinfo['station']}, Chip {self._fileinfo['chip']}, Channel {self._fileinfo['channel']}"
        )
        ax.set_xlabel("ADC")
        ax.set_ylabel("Counts")
        ax.grid("on")

        # Plotting based on arguments
        if scatter == True:
            ax.scatter(self.x, self.y, color="black", marker=".")
            ax.annotate(
                f"From data file:\nTransition point = {self.all_data['t_point']:.2f}\n"
                + f"Width = {self.all_data['width']:.2f}",
                xy=(0.025, 0.975),
                xycoords="axes fraction",
                verticalalignment="top",
                color="black",
                alpha=0.8,
            )

        if show_lin == True:
            ax.plot(self.x_int, lin_y, color="darkturquoise")
            ax.scatter(self.trans_lin, self.half_max, color="darkturquoise", marker="o")
            ax.annotate(
                f"From linear interp.:\nTransition point = {self.trans_lin:.2f}\n",
                xy=(0.025, 0.750),
                xycoords="axes fraction",
                verticalalignment="top",
                color="darkturquoise",
                alpha=0.8,
            )

        if show_erf == True:
            fit_params = [h, t_erf, w_erf]
            erf_x = np.linspace(self.x.min(), self.x.max(), 100)
            erf_y = modified_erf(erf_x, *fit_params)
            ax.plot(erf_x, erf_y, color="darkorange")
            ax.scatter(t_erf, self.half_max, color="darkorange", marker="s")
            ax.annotate(
                f"From erf fit:\nTransition point = {t_erf:.2f} $\pm$ {t_erf_std:.2f}\n"
                + f"Width = {w_erf:.2f} $\pm$ {w_erf_std:.2f}\n",
                xy=(0.025, 0.6),
                xycoords="axes fraction",
                verticalalignment="top",
                color="darkorange",
                alpha=0.8,
            )

        # Plot saving
        if saveplot == True:
            plotname = f"Plot_Claro_Chip{self._fileinfo['chip']}_Ch{self._fileinfo['channel']}.png"
            plt.savefig(plotname, bbox_inches="tight")
            print(f"Plot saved as {os.getcwd()}\{plotname}")

        plt.show()



###############################################################################
#                                Directory analyzer                           #
###############################################################################


class MultiAnalyzer:
    """
    MultiAnalyzer is a class for reading files from a directory path or a .txt file containing a list of file paths.

    Attributes:
    ----------
        path (str): A string containing either the path to a directory or a .txt file containing a list of file paths.

    Methods:
    ----------
        dir_walker_texas_ranger(): Traverse the self.path directory and find all the matching files, storing their paths in a .txt file.
        list_reader(): Read a .txt file containing a list of file paths and return the list of file paths.
        analyzer(discard_unfit=True, savepath=os.getcwd()): Reads self.__file_list, splits the good and bad files and applies the Claro.fit_erf() method to the good files creating .csv file with the results.
    """

    def __init__(self, path):
        """
        The constructor for MultiAnalyzer class.

        Args:
        ----------
            path (str): A string containing either the path to a directory or a .txt file containing a list of file paths.
        """
        self.path = path

    def dir_walker_texas_ranger(self):
        """
        Traverse the self.path directory and find all the matching files.
        Also creates a .txt file containing all the matching file paths.

        Returns:
        ----------
            __file_list (list): A list containing all the full paths of the files.
        """
        top = self.path
        name_to_match = "*Station*_Summary/Chip_*/S_curve/Ch_*_offset_*_Chip_*.txt"
        self.__file_list = []

        for root, dirs, files in os.walk(top):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                if fnmatch.fnmatch(full_path, name_to_match):
                    self.__file_list.append(full_path)

        with open(r".\claro_allfiles.txt", "w") as outfile:
            outfile.write("\n".join(self.__file_list))
        print(f"found {len(self.__file_list)} files to read...")
        print(rf"list of files to analyze created as {os.getcwd()}\claro_allfiles.txt")
        return self.__file_list

    def list_reader(self):
        """
        Reads a .txt file containing a list of file paths.

        Returns:
        ----------
        __file_list (list): A list containing all the full paths of the files.
        """
        self.__file_list = []
        with open(self.path, "r") as all_files:
            self.__file_list = all_files.readlines()
        return self.__file_list

    def analyzer(self, discard_unfit=True, savepath=os.path.abspath(os.getcwd())):
        """
        Reads self.__file_list and splits the good and bad files.
        Applies the Claro.fit_erf() method to the good files and outputs a .csv file with the results. If the fit does not converge, set the erf standard dev to NaN.

        Args:
        ----------
            discard_unfit (bool, optional): If True, puts all the non converging file paths into a "claro_unfit_chips.txt". Defaults to True.
            savepath (string, optional): The save path of the results. Defaults to the current directory.

        Returns:
        ----------
            None
        """
        # Create the savepath folder if it doesn't exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if discard_unfit == True:
            with open(rf"{savepath}\claro_unfit_chips.txt", "w") as create_unfit:
                pass

        _goodfiles = []
        _badfiles = []

        # split good and bad files and write them to files
        for idx, element in enumerate(self.__file_list):
            chip_name = element.strip("\n")
            with open(chip_name, "r") as chip:
                if re.search("[a-zA-Z]", chip.readline()):
                    _badfiles.append(chip_name)
                    continue
                _goodfiles.append(chip_name)
            progress_bar(idx + 1, len(self.__file_list))

        print("\n")
        print(f"found {len(_badfiles)} bad files")
        print(rf"list of bad files created as {savepath}\claro_badfiles.txt")
        with open(rf"{savepath}\claro_badfiles.txt", "w") as outfile:
            outfile.write("\n".join(_badfiles))

        print(f"found {len(_goodfiles)} good files")
        print(rf"list of good files created as {savepath}\claro_goodfiles.txt")
        with open(rf"{savepath}\claro_goodfiles.txt", "w") as outfile:
            outfile.write("\n".join(_goodfiles))

        # read data and evaluate erf t.point from good files
        print("processing the good files...")
        processed_list = []
        for idx, file in enumerate(_goodfiles):
            claro = Claro(file)
            info = claro.get_fileinfo()
            data = claro.get_data()
            erf = claro.fit_erf()

            if discard_unfit == True:
                if np.isnan(erf["transition_point_(erf)"][1]):
                    with open(rf"{savepath}\claro_unfit_chips.txt", "a") as unfit:
                        unfit.write("{}\n".format(data["path"]))
                        continue

            row = [
                info["station"],
                info["chip"],
                info["channel"],
                data["height"],
                data["t_point"],
                data["width"],
                erf["transition_point_(erf)"][0],
                erf["transition_point_(erf)"][1],
            ]
            processed_list.append(row)

            progress_bar(idx + 1, len(_goodfiles))
        print("\n")

        self.processed_df = pd.DataFrame(
            processed_list,
            columns=[
                "Station",
                "Chip",
                "Channel",
                "Amplitude",
                "T_point",
                "Width",
                "erf_t_point",
                "std_erf_t_point",
            ],
        )
        self.processed_df.to_csv(
            rf"{savepath}\claro_processed_chips.csv",
            index=False,
        )

        if discard_unfit == True:
            print(rf"list of unfit files created as {savepath}\claro_unfit_chips.txt")

    def histograms(self, saveplot=True):
        """
        Plot histograms of the transition points, their corresponding erf estimates and the discrepancy between them.

        Parameters:
        ----------
        - saveplot (bool, optional): Indicates whether to save the plot as a png file. Defaults to True

        Returns:
        ----------
        None
        """
        t = np.array(self.processed_df.T_point)
        erf_list = np.array(self.processed_df.erf_t_point)
        discrepancy = t - erf_list


        # plot options
        fig, axs = plt.subplots(3)
        fig.suptitle("Histogram of the transition points distribution")
        [ax.grid("on") for ax in axs]

        axs[0].hist(t, bins=200, color="darkturquoise")
        axs[1].hist(erf_list, bins=200, color="darkorange")
        axs[2].hist(discrepancy, bins=20, range=(-1e-6, 1e-6), color="darkblue")

        axs[0].set_title("Read T. point")
        axs[1].set_title("Erf T. point")
        axs[2].set_title("Discrepancy")

        plt.tight_layout()  # Prevents titles and axes from overlapping

        if saveplot == True:
            plotname = f"Histogram_transition_points.png"
            plt.savefig(plotname, bbox_inches="tight")
            print(f"Plot saved as {os.getcwd()}\{plotname}")
        plt.show()



######################################################################
#           Mathematical functions and other static methods          #
######################################################################


@staticmethod
def progress_bar(progress, total):
    """
    Display a progress bar in the terminal.

    Args:
    ----------
        progress (int): Current progress of the task.
        total (int): Total number of steps in the task.

    Returns:
    ----------
        None
    """
    percent = int(100 * (progress / float(total)))
    bar = "%" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar} | {percent:.2f}%", end="\r")


def modified_erf(x, height, a, b):
    """
    Calculate the modified error function with specified parameters.
    The function returns a shifted error function with height, a, and b as inputs. The shift on the vertical axis is height/2. The input (x-a) is normalized by dividing by b/2 * sqrt(2).

    Args:
    ----------
        x (float): Input value to the error function.
        height (float): Vertical shift of the error function.
        a (float): Horizontal shift of the error function.
        b (float): Scaling factor of the error function.

    Returns:
    ----------
        (float): The modified error function evaluated at x.
    """
    return (height / 2) * (1 + special.erf((x - a) / (b / 2 * np.sqrt(2))))
