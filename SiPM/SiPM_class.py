import pandas as pd
import fnmatch
import os
import sys
import re
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats, signal, optimize


###############################################################################
#                                Single file analyzer                         #
###############################################################################


class Single:
    """
    A class to analyze single ARDU files

    Parameters:
    ----------
    path (str): The file path of the SiPM data file.

    Attributes:
    ----------
        _fileinfo (dict): A dictionary of metadata about the file, including the direction, arduino number,
            test number, and temperature.
        df_sorted (pandas.DataFrame): The data from the .csv file, sorted by SiPM and Step.
        df_grouped (pandas.DataFrame): The sorted data, grouped by SiPM.

    Methods:
    ----------
        get_fileinfo(): Extracts metadata about the file from the file path.
        reader(): Reads the data from the csv file, sorts it, and groups it by SiPM.
        analyzer(): Analyze the SiPM data in either forward or reverse direction and save the results.

    """

    def __init__(self, path):
        """
        Initializes the Single object and sets the path and fileinfo attributes.

        Args:
        ----------
            path (str): The path to the csv file to be analyzed.

        """

        self.path = path
        self.fileinfo = {}
        self.df_grouped = {}

    def get_fileinfo(self):
        """
        Extracts metadata about the file from the file path.

        Returns:
        ----------
            fileinfo (dict): A dictionary of metadata about the file, including the direction, arduino number,
            test number, and temperature.
        """

        path = self.path

        _ardu = re.search(".+ARDU_(.+?)_.+", path).group(1)
        _direction = re.search(".+[0-9]_(.+?)_.+", path).group(1)
        _test = re.search(".+Test_(.+?)_.+", path).group(1)
        _temp = re.search(".+_(.+?)_dataframe.+", path).group(1)

        self.fileinfo = {
            "direction": _direction,
            "ardu": _ardu,
            "test": _test,
            "temp": _temp,
        }
        return self.fileinfo

    def reader(self):
        """
        Reads the data from the csv file, sorts it, and groups it by SiPM.

        Returns:
            df_grouped (pandas.DataFrame): The sorted data, grouped by SiPM.

        """
        path = self.path
        self.get_fileinfo()

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
        return self.df_grouped

    def analyzer(self, room_f_start=0.75, ln2_f_start=1.55, peak_width=10, savepath=os.getcwd(), hide_progress=False):
        """
        Analyze the SiPM data in either forward or reverse direction and save the results.

        Parameters:
        ----------
            room_f_start (float): The starting voltage for room temperature forward analysis. Default is 0.75.
            ln2_f_start (float): The starting voltage for LN2 temperature forward analysis. Default is 1.55.
            peak_width (int): The width of the reverse analysis peak. Default is 10.
            savepath (str): The path to save the results. Default is the current working directory.
            hide_progress (bool): If set to True, progress information will not be printed on terminal. Default is False.

        Returns:
        ----------
            None
        """
        if self.fileinfo["temp"] == "LN2":
            start = ln2_f_start
        else:
            start = room_f_start

        # Create the savepath folder if it doesn't exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # Forward analyzer
        if self.fileinfo["direction"] == "f":
            results = self.df_grouped.apply(fwd_analyzer, start)
            joined_df = self.df_sorted.join(results, on="SiPM")

            out_df = joined_df[["SiPM", "R_quenching", "R_quenching_std"]].drop_duplicates(subset="SiPM")
            res_fname = rf"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_Forward_results.csv"
            out_df.to_csv(os.path.join(savepath, res_fname), index=False)
            if hide_progress is False:
                print(f"Results saved as {savepath}\{res_fname}")

            if hide_progress is False:
                print("Plotting...")
            pdf_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_Forward.pdf"
            pdf_fwd = PdfPages(os.path.join(savepath, pdf_name))
            joined_df.groupby("SiPM").apply(fwd_plotter, pdf_fwd)
            if hide_progress is False:
                print(f"Plot saved as {savepath}\{pdf_name}.")
            pdf_fwd.close()

        # Reverse analyzer
        else:
            results = self.df_grouped.apply(rev_analyzer, peak_width)
            joined_df = self.df_sorted.join(results, on="SiPM")

            out_df = joined_df[["SiPM", "V_bd", "V_bd_std"]].drop_duplicates(subset="SiPM")
            res_fname = rf"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_Reverse_results.csv"
            out_df.to_csv(os.path.join(savepath, res_fname), index=False)
            if hide_progress is False:
                print(f"Results saved as {savepath}\{res_fname}")

            if hide_progress is False:
                print("Plotting...")
            pdf_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_Reverse.pdf"
            pdf_rev = PdfPages(os.path.join(savepath, pdf_name))
            joined_df.groupby("SiPM").apply(rev_plotter, pdf_rev)
            if hide_progress is False:
                print(f"Plot saved as {savepath}\{pdf_name}.")
            pdf_rev.close()



###############################################################################
#                                Directory analyzer                           #
###############################################################################


class DirReader:
    """
    Class for reading and analyzing the SiPM data in a directory.

    The `DirReader` class takes in a directory path and can analyze all the files
    in that directory and its subdirectories that match the pattern "*ARDU_*_dataframe.csv".
    The analysis includes generating histograms of all the R_q and V_bd, with the option to compare temperature and day differences.

    Parameters:
    ----------
        dir (str): The directory path to analyze.

    Attributes:
    ----------
        __file_list (list):  A list containing all the full paths of the files.

    Methods:
    ----------
        dir_walker(): Walk the directory to find all the files that match the correct pattern.
        dir_analyzer(root_savepath = os.getcwd()): Analyze each file in the file list and save the results.
        histograms(compare_temp=True, compare_day=True): Plot histograms of R_q and V_bd.
    """

    def __init__(self, dir):
        """
        Initialize the DirReader object with the directory path.

        Args:
        ----------
            dir (str): the directory path
        """

        self.path = dir

    def dir_walker(self):
        """
        Walk the directory to find all the files that match the pattern "*ARDU_*_dataframe.csv".

        Returns:
        -------
            _file_list (list): list of all the matching file paths (as strings)
        """

        top = self.path
        name_to_match = "*ARDU_*_dataframe.csv"
        self._file_list = []

        for root, dirs, files in os.walk(top):
            for file in files:
                full_path = os.path.join(root, file)
                if fnmatch.fnmatch(full_path, name_to_match):
                    self._file_list.append(full_path)
        return self._file_list

    def dir_analyzer(self, root_savepath=os.getcwd()):
        """
        Analyze each file in the file list and save the results to the root_savepath/results folder.

        Args:
        ----------
            root_savepath (str, optional): the root directory for saving the analysis results, defaults to current working directory.

        Returns:
        ----------
            None
        """

        for idx, file in enumerate(self._file_list):
            try:
                subfolder = re.search(".+\\\\(.+?)\\\\ARDU_.+", file).group(1)
            except AttributeError:
                subfolder = ""

            sipm = Single(file)
            sipm.reader()
            matplotlib.use("Agg")  # Introduced to solve memory issues when dealing with big folders

            sipm.analyzer(
                savepath=os.path.join(root_savepath, "results", subfolder),  # Creates a "results" subdir in the "root_savepath" directory
                hide_progress=True)  # hide_progress set to True to have a cleaner look on the terminal
            progress_bar(idx + 1, len(self._file_list))
        print("\n")

    def histograms(self, compare_temp=True, compare_day=True):
        """
        Plot histograms of R_q and V_bd.

        Args:
        ----------
            compare_temp (bool, optional): If True, produce an histogram that compares the LN2 measures. Defaults to True
            compare_day (bool, optional): If True, produce an histogram that compares the analysis of the 22/23 of April. Defaults to True

        Returns:
        ----------
            None
        """

        top = os.path.join(os.getcwd(), "results")
        fwd_all = []
        rev_all = []

        for subdir, dirs, files in os.walk(top):
            for dir in dirs:
                subdir_path = os.path.join(subdir, dir)

                # Retrieve all the data of fwd and rev
                forward_data = df_join(subdir_path, "Forward")
                reverse_data = df_join(subdir_path, "Reverse")

                if (compare_day == True or compare_temp == True):  # Create the full df only if needed
                    fwd_all.append(forward_data)
                    rev_all.append(reverse_data)

                # Plot R_q and V_bd hist for each subfolder
                fig, axs = plt.subplots(2)
                hist_params(fig, axs, dir)
                forward_data.plot.hist(
                    column=["R_quenching"],
                    ax=axs[0],
                    bins=15,
                    range=(min(forward_data["R_quenching"]), max(forward_data["R_quenching"])),
                    color="darkgreen",
                    alpha=0.7,
                )
                reverse_data.plot.hist(
                    column=["V_bd"],
                    ax=axs[1],
                    bins=15,
                    range=(min(reverse_data["V_bd"]), max(reverse_data["V_bd"])),
                    color="darkorange",
                    alpha=0.7,
                )
                plt.tight_layout()  # Prevents titles and axes from overlapping
                plotname = f"Histograms_{dir}.png"
                plt.savefig(os.path.join(top, plotname), bbox_inches="tight")
                plt.close()
                print(f"Plot saved as {top}\{plotname}")

        if compare_temp == True or compare_day == True:
            # Merge all the fwd and rev dataframes
            fwd_all = pd.concat(fwd_all)
            rev_all = pd.concat(rev_all)

        if compare_temp == True:
            ln2_fwd = fwd_all[fwd_all["subdir"].str.contains("LN2")]
            ln2_rev = rev_all[fwd_all["subdir"].str.contains("LN2")]

            fig, axs = plt.subplots(2)
            hist_params(fig, axs, "Liquid Nitrogen comparison")
            for subdir, group in ln2_fwd.groupby("subdir"):
                group["R_quenching"].hist(ax=axs[0], label=subdir, bins=15, alpha=0.6)
            for subdir, group in ln2_rev.groupby("subdir"):
                group["V_bd"].hist(ax=axs[1], label=subdir, bins=15, alpha=0.6)
            [ax.legend() for ax in axs]
            plt.tight_layout()
            plotname = f"LN2_comparison_hist.png"
            plt.savefig(os.path.join(top, plotname), bbox_inches="tight")
            plt.close()
            print(f"Plot saved as {top}\{plotname}")

        if compare_day == True:
            fwd_april = fwd_all[fwd_all["subdir"].str.contains("_04_")]
            rev_april = rev_all[fwd_all["subdir"].str.contains("_04_")]

            fig, axs = plt.subplots(2)
            hist_params(fig, axs, "April data comparison")
            for subdir, group in fwd_april.groupby("subdir"):
                group["R_quenching"].hist(ax=axs[0], label=subdir, bins=15, alpha=0.6)
            for subdir, group in rev_april.groupby("subdir"):
                group["V_bd"].hist(ax=axs[1], label=subdir, bins=15, alpha=0.6)
            [ax.legend() for ax in axs]
            plt.tight_layout()
            plotname = f"April_data_comparison_hist.png"
            plt.savefig(os.path.join(top, plotname), bbox_inches="tight")
            plt.close()
            print(f"Plot saved as {top}\{plotname}")



######################################################################
#           Mathematical functions and other static methods          #
######################################################################


@staticmethod
def fwd_analyzer(data, starting_point):
    """
    Analyze the forward IV Curve by fitting a line on the linear part of the data.

    Args:
    ----------
        data (pandas.DataFrame): pd DataFrame containing the values of V, I and I_err.
        starting_point (float): specifies the starting point from where to isolate the linear data.

    Returns:
    ----------
        values (pandas.Series): A pandas Series containing the following values:
            R_quenching (float): the quenching resistance of the system calculated using linear regression.
            R_quenching_std (float), an overestimation of the standard deviation of R_quenching.
            start (float), the starting point used to isolate the linear data.
            m (float), the slope of the regression line.
            q (float), the intercept of the regression line.
    """
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
    R_quenching_std = max(model.stderr, 0.03 * R_quenching)  # overestimation of the R standard dev

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
    """
    Plots the data and results of the forward IV curve.

    Args:
    ----------
        data (pandas.DataFrame): pd DataFrame containing the metadata, data and regression results for each SiPM analysis.
        pdf ( matplotlib.backends.backend_pdf.PdfPages): A PdfPages object used to save the pdf.

    Returns:
    ----------
    None
    """

    data["y_lin"] = (data["m"] * data["V"] + data["q"])  # Find y values via linear regression
    lin_x = data[data["V"] >= data["start"]]["V"]  # The conditions are there to plot only on the linear part of the curve
    lin_y = data[data["V"] >= data["start"]]["y_lin"]

    fig, ax = plt.subplots()
    sipm_number = list(data["SiPM"].drop_duplicates())[0]
    fig.suptitle(f"Forward IV curve: SiPM {sipm_number}")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current(mA)")
    ax.grid("on")

    ax.errorbar(data["V"], data["I"], data["I_err"], marker=".", zorder=1)
    ax.plot(
        lin_x,
        lin_y,
        color="darkorange",
        linewidth=1.2,
        label=f'Linear fit: Rq = ({data["R_quenching"].iloc[0]:.2f} $\pm$ {data["R_quenching_std"].iloc[0]:.2f}) $\Omega$',
        zorder=2,
    )

    ax.legend(loc="upper left")
    pdf.savefig()
    plt.close()


@staticmethod
def rev_analyzer(data, peak_width):
    """
    Analyze the reverse IV Curve by fitting a 5th-degree polynomial on the data derivative and a gaussian curve on the poly peak.

    Args:
    ----------
        data (pandas.DataFrame): pd DataFrame containing the values of V, I and I_err.
        peak_width (int): The width of the peak to search.

    Returns:
    ----------
        values (pandas.Series): A pandas Series containing the following values:
            V_bd (float): The breakdown voltage, evaluated as the mean of the gaussian curve.
            V_bd_std (float): The breakdown voltage standard deviation, evaluated as the standard deviation of the gaussian curve.
            width (float): the FWHM of the gaussian curve.
            coefs (list) : list of the 5th-degree polynomial coefficients.
            params (list): the parameters of the curve fit of the gaussian.
    """

    x = data["V"].to_numpy()
    y = data["I"].to_numpy()

    # Evaluation of the 1st derivative
    derivative = norm_derivative(x, y)

    # 5th degree polynomial fit
    fifth_poly = Polynomial.fit(x, derivative, 5)
    coefs = fifth_poly.convert().coef
    y_fit = fifth_poly(x)

    # Peak finder
    peaks = signal.find_peaks(fifth_poly(x), width=peak_width)[0]  # width parameter to discard smaller peaks
    idx_max = peaks[np.argmax(fifth_poly(x)[peaks])]
    x_max = x[idx_max]
    fwhm = x[int(idx_max + peak_width / 2)] - x[int(idx_max - peak_width / 2)]
    
    # Gaussian fit around the peak
    x_gauss = x[np.logical_and(x >= (x_max - fwhm / 2), x <= (x_max + fwhm / 2))]
    y_gauss = y_fit[np.logical_and(x >= (x_max - fwhm / 2), x <= (x_max + fwhm / 2))]

    fit_guess = [0, 1, x_max, fwhm/2]
    params, covar = optimize.curve_fit(gauss, x_gauss, y_gauss, fit_guess, maxfev=20000)

    # Returning the values
    values = pd.Series(
        {
            "V_bd": params[2],
            "V_bd_std": params[3],
            "width": fwhm,
            "coefs": coefs,
            "params": params,
        }
    )
    return values


@staticmethod
def rev_plotter(data, pdf):
    """Plots the data and results of the forward IV curve.

    Args:
    ----------
        data (pandas.DataFrame): pd DataFrame containing the metadata, data and fit results for each SiPM analysis.
        pdf ( matplotlib.backends.backend_pdf.PdfPages): A PdfPages object used to save the pdf.

    Returns:
    ----------
    None
    """

    x = data["V"].to_numpy()
    y = data["I"].to_numpy()

    V_bd = data["V_bd"].iloc[0]
    poly_coefs = data["coefs"].iloc[0]

    derivative = norm_derivative(x, y)
    y_poly = (
        poly_coefs[0]
        + poly_coefs[1] * x
        + poly_coefs[2] * x**2
        + poly_coefs[3] * x**3
        + poly_coefs[4] * x**4
        + poly_coefs[5] * x**5
    )
    x_gauss = x[np.logical_and(x >= (V_bd - data["width"].iloc[0] / 2), x <= (V_bd + data["width"].iloc[0] / 2))]
    y_gauss = gauss(x_gauss, *data["params"].iloc[0])

    fig, ax = plt.subplots()
    sipm_number = list(data["SiPM"].drop_duplicates())[0]
    fig.suptitle(f"Reverse IV curve: SiPM {sipm_number}")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current(mA)")
    ax.grid("on")

    ax.set_yscale("log")
    ax.errorbar(data["V"], data["I"], data["I_err"], marker=".", label="Data")
    ax.legend(loc="upper right")

    ax2 = ax.twinx()
    ax2.tick_params(axis="y", colors="darkgreen")

    ax2.set_ylabel(r"$I^{-1} \frac{dI}{dV}$", color="darkgreen")
    ax2.scatter(x, derivative, marker="o", s=5, color="darkgreen", label="Derivative")
    ax2.plot(x, y_poly, color="darkturquoise", label="5th-deg polynomial")
    ax2.plot(x_gauss, y_gauss, color="darkorange", label="Gaussian around peak")
    ax2.axvline(V_bd, color="gold", label=f"$V_{{Bd}}$ = {V_bd:.2f} $\pm$ {abs(data['V_bd_std'].iloc[0]):.2f} V")
    ax2.legend(loc="upper left")

    pdf.savefig()
    plt.close()


@staticmethod
def df_join(directory, direction):
    """
    A static method to join multiple CSV files into one Pandas DataFrame.
    
    Args:
    ----------
    directory (str): the directory of the data to retrieve.
    direction (str): string to match
    
    Returns:
    ----------
    data (pandas.Dataframe): A Pandas DataFrame containing the contents of the joined csv files and an additional column with the name of the directory.
    """
    
    files = [file for file in os.listdir(directory) if direction in file and file.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join(directory, file)) for file in files]
    data = pd.concat(dfs)
    data["subdir"] = os.path.basename(directory)
    return data


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


# Gaussian curve to fit
def gauss(x, H, A, mu, sigma):
    """
    Returns a gaussian curve with displacement H, amplitude A, mean mu and standard deviation sigma.

    Parameters:
    ----------
        x (numpy.ndarray): Input values for the curve.
        H (float): The displacement of the curve.
        A (float): The amplitude of the curve.
        mu (float): The mean of the curve.
        sigma (float): The standard deviation of the curve.

    Returns:
    ----------
        numpy.ndarray: The values of the gaussian curve for the input x.
    """
    return H + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# Normalized derivative
def norm_derivative(x, y):
    """
    Evaluate the normalized derivative of the x and y data as 1/y * dy/dx.

    Args:
        x (np.array): data on the x axis
        y (np.array): data on the y axis

    Returns:
        float : the normalized derivative
    """
    dy_dx = np.gradient(y) / np.gradient(x)
    return 1 / y * dy_dx


# Title and axis labels for the histograms
def hist_params(fig, axs, dir):
    """
    This function takes as input a figure and two axes (as a list) and sets the titles, labels and grids for the histograms.
    The first histogram is for the quenching resistance and the second one is for the breakdown voltage.

    Args:
        fig (matplotlib.figure): Figure to set the title of
        axs (list): lsit of the axes of the plot
        dir (str): direction of the IV curve
    """
    fig.suptitle(f"{dir}: R_q and V_Bd distribution")
    [ax.grid("on") for ax in axs]
    axs[0].set_title("Quenching Resistance Histogram")
    axs[0].set_xlabel("$R_q [\Omega$]")
    axs[0].set_ylabel("Frequency")
    axs[1].set_title("Breakdown Voltage Histogram")
    axs[1].set_xlabel("$V_{Bd}$ [V]")
    axs[1].set_ylabel("Frequency")
