import os
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
        self.filename = path
        data = pd.read_csv(self.filename, sep='\t', header=None, skiprows=None)
        self.height= data[0][0]
        self.t_point = data[1][0]
        self.width = np.abs(data[2][0])
        self.x = data.iloc[2:,0].to_numpy()
        self.y = data.iloc[2:,1].to_numpy()

        self._metadata ={'path': self.filename,
                         'amplitude': self.height,
                         'transition point': self.t_point,
                         'width': self.width,
                        }

        
        self.values ={'slope': None,
                      'intercept': None,
                      'transition point (Linear)': None
                     }
        self.fit_guess  = [ self.height, self.t_point, self.width ]




    # linear fit method
    def fit_lin(self):
        # vector parsing (allows for base and saturation values to be different than 0 and 1000)
        xInt = self.x
        yInt = self.y
        while(yInt[0]==yInt[1]):
            xInt=np.delete(xInt,0)
            yInt = np.delete(yInt,0)
        while (yInt[-1]==yInt[-2]):
            xInt=np.delete(xInt,-1)
            yInt = np.delete(yInt,-1)

        # linear regression
        model = stats.linregress(xInt,yInt)

        # transition point evaluation
        halfMaximum = (yInt[-1] - yInt[0])/2
        transLin = (halfMaximum - model.intercept)/model.slope

        # saving the values
        values = {'slope': model.slope,
                  'intercept': model.intercept,
                  'transition point (Linear)': transLin}
        return values
        


    # erf fit method
    def fit_erf(self,  guesses:list='default'):
        function = Claro.modified_erf
        x = self.x
        y = self.y
        if guesses=='default':
            guesses=self.fit_guess
        params, covar = optimize.curve_fit(function, x, y, p0=guesses)
        erf_fit_x = np.linspace(self.x.min(), self.x.max(), 100)
        erf_fit_y= function(erf_fit_x , *params)
        return erf_fit_x, erf_fit_y

    # data printing method
    def printData(self):
        for key, value in self._metadata.items():
            print(key, ': ', value)
        for key, value in self.fit_lin().items():
            print(key, ': ', value)



    # plotter method
    def plotter(self):
        pass





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
        