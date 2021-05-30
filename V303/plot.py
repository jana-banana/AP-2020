import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

#AUFGABE 1 ohne Noise generator
print('AUFGABE 1')

phi, A_ohne, A_mit= np.genfromtxt('data/data_ohne_mit.txt', unpack=True)

phi *= np.pi/180

#def U(phi, U0, B):
#    return 2/np.pi* U0 * np.cos(phi+B)

def U(phi, U_0, kA):
    return abs(U_0 * np.cos(phi))

par, covm = curve_fit(
    U,
    phi,
    A_ohne,
    sigma=None,
    absolute_sigma=True,
    p0=[10, 1]
    )
err = np.sqrt(np.diag(covm))

plt.plot(phi , A_ohne , '.', label = 'Messwerte ohne')
x_plot = np.linspace(0, 2*np.pi, 1000)
plt.xlabel('$\phi$')
plt.ylabel('Spannung $U$ [V]')

plt.plot(x_plot,U(x_plot, *par) , 'r-', label="Nicht-Lineare Regression")
#plt.plot(phi, line(phi,),'r-', label='Ausgleichgerade')
plt.grid()
plt.tight_layout()
plt.legend()
#plt.savefig('content/abbildungen/plot1.pdf')

plt.clf()

#AUFGABE 1 mit Noise generator

plt.plot(phi , A_mit , '.', label = 'Messwerte mit')
plt.xlabel('$\phi$')
plt.ylabel('Spannung $U$ [V]')

plt.grid()
plt.tight_layout()
plt.legend()
#plt.savefig('content/abbildungen/plot2.pdf')

plt.clf()

#AUFGABE 2 

r, A_dioden= np.genfromtxt('data/dioden.txt', unpack=True)

#r = np.array([r_1])
#A_dioden = np.array([A_dioden_1])

plt.plot(r[1:], A_dioden[1:],'+', label='Fit')
#plt.plot(phi , A_dioden , '.', label = 'Messwerte')
plt.xlabel('r /cm')
plt.ylabel('Spannung $U$ [V]')

def line_dioden(r, a, b):
    return a* 1/(r) + b

popt, pcov = curve_fit(line_dioden, r, A_dioden)
errors = np.sqrt(np.diag(pcov))
print("a =", popt[0])
print("a fehler =",errors[0])
print("b =", popt[1])
print("b fehler=",errors[1])

plt.plot(r, line_dioden(r, popt[0], popt[1]),'r-', label='Ausgleichgerade')

#plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('content/abbildungen/plot3.pdf')

plt.clf()