import os
import re
import pandas as pd
import numpy as np
from scipy import optimize , special , stats
import matplotlib.pyplot as plt

###############################################################################
#                                Single file analyzer                         #
###############################################################################
class Single():

    # Constructor definition
    def __init__(self, path):
        self.path = path
        data = pd.read_csv(self.path, sep='\t', header=None, skiprows=None)
        self.height= data[0][0]
        self.t_point = data[1][0]
        self.width = np.abs(data[2][0])
        self.x = data.iloc[2:,0].to_numpy()
        self.y = data.iloc[2:,1].to_numpy()

        
        self._metadata ={'path': self.path,
                         'amplitude': self.height,
                         'transition point': self.t_point,
                         'width': self.width,
                        }

        self.fit_guess  = [ self.height, self.t_point, self.width ]


    # Method to retrieve Station, Chip and Channel number
    def __get_fileinfo(self,path):
        try:
            _chip = re.search('Chip_(.+?).txt' , path).group(1)
            _channel = re.search('Ch_(.+?)_' , path).group(1)
            _station = re.search('\Station_(.+?)__' , path).group(1)
        except AttributeError:
            _station = '?'

        _fileinfo ={'station' : _station,
                         'chip' : _chip,
                         'channel' : _channel
                    }
        return _fileinfo



    # linear fit method
    def fit_lin(self):
        # vector parsing (allows for base and saturation values to be different than 0 and 1000)
        x_int = self.x
        y_int = self.y
        while(y_int[0]==y_int[1]):
            x_int=np.delete(x_int,0)
            y_int = np.delete(y_int,0)
        while (y_int[-1]==y_int[-2]):
            x_int=np.delete(x_int,-1)
            y_int = np.delete(y_int,-1)
        self.x_int = x_int
        self.y_int = y_int
        
        # linear regression and t-point eval
        model = stats.linregress(x_int,y_int)

        half_maximum = (y_int[-1] - y_int[0])/2
        trans_lin = (half_maximum - model.intercept)/model.slope
        self.half_max = half_maximum
        self.trans_lin = trans_lin

        # saving the values
        values = {'slope': model.slope,
                  'intercept': model.intercept,
                  'transition point (Linear)': trans_lin,
                  'R_squared' : model.rvalue**2}
        return values
        


    # erf fit method
    def fit_erf(self):
        function = Claro.modified_erf
        x = self.x
        y = self.y

        params, covar = optimize.curve_fit(function, x, y, self.fit_guess)

        erf_fit_x = np.linspace(self.x.min(), self.x.max(), 100)
        erf_fit_y= function(erf_fit_x , *params)
        self.erf_x = erf_fit_x
        self.erf_y = erf_fit_y
        std = np.sqrt(np.diag(covar))
        
        self.erf_params = {
                'height' : [params[0], std[0]],
                'transition point (erf)' : [params[1], std[1]],
                'width' : [params[2], std[2]]
            }
        return self.erf_params




    # data printing method
    def printData(self):
        for key, value in self._metadata.items():
            print(key, ': ', value)
        print ('\n')
        for key, value in self.fit_lin().items():
            print(key, ': ', value)
        for key, value in self.fit_erf().items():
            print(key, ': ', value)
        print ('\n')
        


    # plotter method
    def plotter(self, scatter = True, show_lin = True, show_erf = True, t_lin = True, t_erf = True):

        # retrieving the data to be plotted
        fileinfo = self.__get_fileinfo(self.path)
        x = self.x
        y = self.y
        x_int = self.x_int
        lin_intercept =self.fit_lin().get('intercept')
        lin_slope = self.fit_lin().get('slope')
        lin_y = lin_slope*x_int + lin_intercept
        erf_x = self.erf_x
        erf_y = self.erf_y

        # plot options
        fig, ax = plt.subplots()
        fig.suptitle(f"Fit CLARO: Station {fileinfo['station']}, Chip {fileinfo['chip']}, Channel {fileinfo['channel']}")
        #ax.errorbar(x, y, y_err, marker = '.', label = 'Data', zorder = 1)
        ax.set_xlabel("ADC")
        ax.set_ylabel("Counts")

        # plotting based on arguments
        if scatter == True:
            plt.scatter(x, y, color = 'black', marker ='o')
        plt.plot(x_int, lin_y, color = 'green')
        plt.plot(erf_x, erf_y, color = 'blue')

        plt.grid("on")
        plt.show()
        





###############################################################################
#                                Directory analyzer                           #
###############################################################################

class Claro():

    ######################################################################
    #           Mathematical functions and other static methods          #
    ######################################################################
    
    @staticmethod
    def modified_erf(x, height, a, b):
        """Returns a modified erf function, shifted on the vertical
        axis by height/2, with parameters a and b.
        """
        return (height/2)*(1+special.erf((x-a)/(b/2*np.sqrt(2))))
        
