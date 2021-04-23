import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import os

if os.path.exists("build") == False:
    os.mkdir("build")

#curve-fit
#params_vi , ma_vi = np.polyfit(U_vi,np.sqrt(I_vi), deg =1, cov = True)
#errors_vi = np.sqrt(np.diag(ma_vi))

