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
        return self.df_grouped

    # Analyzer and plotter method
    def analyzer(
        self,
        room_f_start=0.75,
        ln2_f_start=1.55,
        peak_width=10,
        savepath=os.getcwd(),
        hide_progress=False,
    ):
        width = peak_width
        if self._fileinfo["temp"] == "LN2":
            start = ln2_f_start
        else:
            start = room_f_start

        # Create the savepath folder if it doesn't exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # Forward analyzer
        if self._fileinfo["direction"] == "f":
            results = self.df_grouped.apply(fwd_analyzer, start)
            joined_df = self.df_sorted.join(results, on="SiPM")


            out_df = joined_df[
                ["SiPM", "R_quenching", "R_quenching_std"]
            ].drop_duplicates(subset="SiPM")
            res_fname = rf"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Temp{self._fileinfo['temp']}_Forward_results.csv"
            out_df.to_csv(os.path.join(savepath, res_fname), index=False)
            if hide_progress is False:
                print(f"Results saved as {savepath}\{res_fname}")

            if hide_progress is False:
                print("Plotting...")
            pdf_name = f"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Temp{self._fileinfo['temp']}_Forward.pdf"
            pdf_fwd = PdfPages(os.path.join(savepath, pdf_name))
            joined_df.groupby("SiPM").apply(fwd_plotter, pdf_fwd)
            if hide_progress is False:
                print(f"Plot saved as {savepath}\{pdf_name}.")
            pdf_fwd.close()

        # Reverse analyzer
        else:
            results = self.df_grouped.apply(rev_analyzer, width)
            joined_df = self.df_sorted.join(results, on="SiPM")


            out_df = joined_df[["SiPM", "V_bd", "V_bd_std"]].drop_duplicates(
                subset="SiPM"
            )
            res_fname = rf"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Temp{self._fileinfo['temp']}_Reverse_results.csv"
            out_df.to_csv(os.path.join(savepath, res_fname), index=False)
            if hide_progress is False:
                print(f"Results saved as {savepath}\{res_fname}")


            if hide_progress is False:
                print("Plotting...")
            pdf_name = f"Arduino{self._fileinfo['ardu']}_Test{self._fileinfo['test']}_Temp{self._fileinfo['temp']}_Reverse.pdf"
            pdf_rev = PdfPages(os.path.join(savepath, pdf_name))
            joined_df.groupby("SiPM").apply(rev_plotter, pdf_rev)
            if hide_progress is False:
                print(f"Plot saved as {savepath}\{pdf_name}.")
            pdf_rev.close()



###############################################################################
#                                Directory analyzer                           #
###############################################################################


class DirReader:

    def __init__(self, dir):
        self.dir = dir
        self.path = dir
        self.__file_list = []


    def dir_walker(self):
        top = self.path
        name_to_match = "*ARDU_*_dataframe.csv"
        file_list = []

        for root, dirs, files in os.walk(top):
            for file in files:
                full_path = os.path.join(root, file)
                if fnmatch.fnmatch(full_path, name_to_match):
                    file_list.append(full_path)
        self.__file_list = file_list
        return self.__file_list


    def dir_analyzer(self , root_savepath = os.getcwd()):
        for idx, file in enumerate(self.__file_list):
            
            try:
                subfolder = re.search(".+\\\\(.+?)\\\\ARDU_.+", file).group(1)
            except AttributeError:
                subfolder = ""
            print(subfolder)

            sipm = Single(file)
            sipm.reader()
            matplotlib.use(
                "Agg"
            )  # Introduced to solve memory issues when dealing with big folders

            sipm.analyzer(
                savepath=os.path.join(root_savepath, "results", subfolder), # Creates a "results" subdir in the "root_savepath" directory
                hide_progress=True,
            )  # hide_progress set to True to have a cleaner look on the terminal
            progress_bar(idx + 1, len(self.__file_list))
        print("\n")

    def histograms(self, compare_temp=True, compare_day=True):
        top = os.path.join(os.getcwd(), "results")
        fwd_all = []
        rev_all = []

        for subdir, dirs, files in os.walk(top):
            for dir in dirs:
                subdir_path = os.path.join(subdir, dir)

                # Retrieve all the data of fwd and rev
                forward_data = df_join(subdir_path, "Forward")
                reverse_data = df_join(subdir_path, "Reverse")

                if (
                    compare_day == True or compare_temp == True
                ):  # Create the full df only if needed
                    fwd_all.append(forward_data)
                    rev_all.append(reverse_data)

                # Plot R_q and V_bd hist for each subfolder
                fig, axs = plt.subplots(2)
                hist_params(fig, axs, dir)
                forward_data.plot.hist(
                    column=["R_quenching"],
                    ax=axs[0],
                    bins=15,
                    range=(
                        min(forward_data["R_quenching"]),
                        max(forward_data["R_quenching"]),
                    ),
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
    data["y_lin"] = (
        data["m"] * data["V"] + data["q"]
    )  # Find y values via linear regression
    lin_x = data[data["V"] >= data["start"]][
        "V"
    ]  # The conditions are there to plot only on the linear part of the curve
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
        zorder=2,
    )
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
def rev_analyzer(data, peak_width):
    x = data["V"].to_numpy()
    y = data["I"].to_numpy()

    # Evaluation of the 1st derivative
    derivative = norm_derivative(x, y)

    # 5th degree polynomial fit
    fifth_poly = Polynomial.fit(x, derivative, 5)
    coefs = fifth_poly.convert().coef
    y_fit = fifth_poly(x)

    # Peak finder
    peaks = signal.find_peaks(fifth_poly(x), width=peak_width)[
        0
    ]  # width parameter to discard smaller peaks
    idx_max = peaks[np.argmax(fifth_poly(x)[peaks])]
    x_max = x[idx_max]
    fwhm = x[int(idx_max + peak_width / 2)] - x[int(idx_max - peak_width / 2)]
    # Gaussian fit around the peak
    x_gauss = x[
        np.logical_and(
            x >= (x_max - fwhm / 2),
            x <= (x_max + fwhm / 2),
        )
    ]
    y_gauss = y_fit[
        np.logical_and(
            x >= (x_max - fwhm / 2),
            x <= (x_max + fwhm / 2),
        )
    ]

    fit_guess = [0, 1, x_max, fwhm]
    params, covar = optimize.curve_fit(gauss, x_gauss, y_gauss, fit_guess, maxfev=10000)
    mu = params[2]
    std = params[3]

    # Returning the values
    values = pd.Series(
        {
            "V_bd": mu,
            "V_bd_std": std,
            "width": fwhm,
            "coefs": coefs,
            "params": params,
        }
    )
    return values


@staticmethod
def rev_plotter(data, pdf):
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
    x_gauss = x[
        np.logical_and(
            x >= (V_bd - data["width"].iloc[0] / 2),
            x <= (V_bd + data["width"].iloc[0] / 2),
        )
    ]
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
    ax2.axvline(
        V_bd,
        color="gold",
        label=f"$V_{{Bd}}$ = {V_bd:.2f} $\pm$ {abs(data['V_bd_std'].iloc[0]):.2f} V",
    )
    ax2.legend(loc="upper left")

    pdf.savefig()
    plt.close()


@staticmethod
def norm_derivative(x, y):
    dy_dx = np.gradient(y) / np.gradient(x)
    return 1 / y * dy_dx


@staticmethod
def df_join(directory, direction):
    files = [
        file
        for file in os.listdir(directory)
        if direction in file and file.endswith(".csv")
    ]
    dfs = [pd.read_csv(os.path.join(directory, file)) for file in files]
    data = pd.concat(dfs)
    data["subdir"] = os.path.basename(directory)
    return data


@staticmethod
def progress_bar(progress, total):
    """Provides a visual progress bar on the terminal"""

    percent = int(100 * (progress / float(total)))
    bar = "%" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar} | {percent:.2f}%", end="\r")


# Gaussian curve to fit
def gauss(x, H, A, mu, sigma):
    # Returns a gaussian curve with displacement H, amplitude A, mean mu and std sigma.
    return H + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# Title and axis labels for the histograms
def hist_params(fig, axs, dir):
    fig.suptitle(f"{dir}: R_q and V_Bd distribution")
    [ax.grid("on") for ax in axs]
    axs[0].set_title("Quenching Resistance Histogram")
    axs[0].set_xlabel("$R_q [\Omega$]")
    axs[0].set_ylabel("Frequency")
    axs[1].set_title("Breakdown Voltage Histogram")
    axs[1].set_xlabel("$V_{Bd}$ [V]")
    axs[1].set_ylabel("Frequency")
