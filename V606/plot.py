import matplotlib.pyplot as plt
import numpy as np

f, U_A = np.genfromtxt('data_selektiv.txt', unpack=True)

U = U_A/(1000)
plt.plot(f,U, label='Messwerte')
#plt.xlabel(r'$\nu \:/\: \si{\kilo\hertz}$')
#plt.ylabel(r'$\frac{U_{\text{A}}}{U_{\text{E}}}$')
plt.legend()


# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout()
plt.savefig('filterkurve.pdf')
