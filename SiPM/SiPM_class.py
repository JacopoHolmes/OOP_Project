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

class Single():

    # Constructor definition
    def __init__(self, path):
        self.path = path
        self._fileinfo = {}



    # fileinfo retriever method
    def _get_fileinfo(self):
        path = self.path

        _ardu = re.search('.+ARDU_(.+?)_.+' , path).group(1)
        _direction = re.search('.+[0-9]_(.+?)_.+' , path).group(1)
        _number = re.search('.+Test_(.+?)_.+' , path).group(1)
        _temp = re.search('.+_(.+?)_dataframe.+' , path).group(1)

        self._fileinfo = {'direction' : _direction,
                    'ardu' : _ardu,
                    'number' : _number,
                    'temp' : _temp
                   }
        return self._fileinfo
    


    # Data reader method
    def reader(self):
        path = self.path
        self._get_fileinfo()

        # Skip rows until header (useful if the file contains comments on top)
        def header_finder(path):
            with open (path , 'r') as file:
                for num, line in enumerate(file, 0):
                    if 'SiPM' in line:
                        return num
                    else:
                        print ("Error : header not found")
                        sys.exit(1)

        df = pd.read_csv(path , header = header_finder(path))
        df_sorted = df.sort_values(by=['SiPM', 'Step'], ignore_index=True)
        print(df_sorted)
        self.sdf = df_sorted
    


    def analyzer(self , f_starting_point = 1.55):
        start = f_starting_point
        if self._fileinfo['direction'] == 'f':
            df_grouped = self.sdf.groupby("SiPM")
            results = df_grouped.apply(fwd_analyzer, start)
            joined_df = self.sdf.join(results , on = 'SiPM')
            print(joined_df)
            #result = [fwd_analyzer(group , f_starting_point) for sipm, group in df_grouped]
            pdf_fwd = PdfPages(f"Arduino {self._fileinfo['ardu']} Number {self._fileinfo['ardu']} Forward.pdf")
            joined_df.groupby('SiPM').apply(fwd_plotter , pdf_fwd)
            pdf_fwd.close()


        
        else:
            pdf_rev = PdfPages(f"Arduino {self._fileinfo['ardu']} Number {self._fileinfo['ardu']} Reverse.pdf")
            pdf_rev.close()
            pass



@staticmethod
def fwd_analyzer(data , starting_point):

    x = data['V']
    y = data['I']
    # isolate the linear data
    x_lin = x[x >= starting_point]
    y_lin = y[x >= starting_point]

    # linear regression
    model = stats.linregress(x_lin , y_lin)
    m = model.slope
    q = model.intercept
    R_quenching = 1000/m
    R_quenching_std = max(model.stderr, 0.03*R_quenching)

    # saving the values
    values = pd.Series({ 'R_quenching' :  R_quenching, 'R_quenching_std' : R_quenching_std, 'start' : starting_point ,  'm' : m , 'q' : q})
    return values



def fwd_plotter(data, pdf):

    data['y_lin'] = (data['m']*data['V'] + data['q'])

    

    fig , ax = plt.subplots()
    fig.suptitle("Forward IV curve")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current(mA)")
    ax.grid("on")

    ax.errorbar(data['V'] , data['I'] ,  data['I_err'] , marker = '.' ,zorder = 1)
    ax.plot(data[data['V']>= data['start']]['V'] , data[data['V']>= data['start']]['y_lin'] , color = 'green' , linewidth = 1.2 , zorder = 2)
    ax.annotate(f'Linear fit: Rq = ({data["R_quenching"].iloc[0]:.2f} $\pm$ {data["R_quenching_std"].iloc[0]:.2f}) $\Omega$',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                verticalalignment='top', color='black')

    pdf.savefig()




###############################################################################
#                                Directory analyzer                           #
###############################################################################

class dir_reader():

    # Constructor definition
    def __init__(self, dir):
        self.dir = dir

    # File finder method
    def dir_walker(self):
            top = self.path
            name_to_match= 'ARDU_*_dataframe.csv'
            file_list = []
            
            for root, dirs, files in os.walk(top):
                for file in files:
                    full_path = os.path.join(root, file)
                    if fnmatch.fnmatch(full_path , name_to_match):
                        file_list.append(full_path)

            self.__file_list = file_list
            return self.__file_list



