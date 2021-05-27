import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

import os

if os.path.exists("build") == False:
    os.mkdir("build")


###Schallgeschwindigkeit ausrechnen
Nr, t, A, y = np.genfromtxt('data/schall.txt', unpack=True)
t_plot = np.linspace(0, 5e-5, 1000)

y*=1e-3
t*=1e-6


params, covariance_matrix = np.polyfit(t, y, deg =1 , cov = True)
errors = np.sqrt(np.diag(covariance_matrix))

print('a', params[0], errors[0])
print('c', params[1], errors[1])

c = 2 * ufloat(params[0], errors[0]) #
c_theo = 2730

print('c', c)
print('diff', 1-(c_theo/c))

plt.figure()
plt.plot(t, y, 'k.', label='Messwerte')
plt.plot(t_plot, params[0]*t_plot + params[1], 'r-', label='Ausgleichsgerade')
plt.xlabel(r'$ t \quad / \quad \si{\second}$')
plt.ylabel(r'$ s \quad / \quad \si{\metre}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/schallgeschwindigkeit.pdf')

s = c_theo * t /2
print('s', s)
print('diff', s-y)
print('prozente',(1 - s/y)*10**2 )


###Absorptionskoeffizient

#Ich will mit Zeiten rechnen
y_2 = c * t #metre
y_2plot = np.linspace(0.01, 0.13, 1000)

def I(x, I_0, a):
    return I_0 * np.exp(-a * x)

parameter , cov_matrix = curve_fit(I , unp.nominal_values(y_2) , A )
err = np.sqrt(np.diag(cov_matrix))

print('I_O=', parameter[0], err[0])
print('a =', parameter[1], err[1])

plt.figure()
plt.plot(unp.nominal_values(y_2) , A, 'k.', label='Messwerte')
plt.plot(y_2plot, I(y_2plot, *parameter), 'r-', label='curve_fit')
plt.xlabel(r'$s \quad / \quad \si{\metre}$')
plt.ylabel(r'$ I \quad / \quad \si{\volt}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/absorptionskoeff.pdf')

###Auge

c_linse = 2500 #metre\per\second
c_glaskörper = 1419 #metre\per\second

t1 = 11.3e-6
t2 = 17.4e-6
t3 = 24.6e-6

S_I = t1 * c_glaskörper / 2
S_L = S_I + (t2-t1)*c_linse/2 
S_R = S_L + (t3-t2)*c_glaskörper/2

print('S_Iris', S_I, 'm', S_I*10**3, 'mm' )
print('S_linse', S_L, 'm', S_L*10**3, 'mm' )
print('S_retina', S_R, 'm', S_R*10**3, 'mm' )