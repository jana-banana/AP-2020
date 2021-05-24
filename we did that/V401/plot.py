import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)

#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')

#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')

#Spiegel verschieben
print('Spiegel verschieben')
del_d, impulse = np.genfromtxt('data_lambda.txt', unpack=True)

z = unp.uarray(impulse, 4) #Impulse mit Fehler von 4
print(z)
ü = 5.046 #Hebelübersetzung

wl = 2 * del_d / (ü*z) #Millimeter
print(wl*10**6) #Nanometer für Auswertung

wl_mittel = np.mean(wl)
print(wl_mittel*10**6) #Nanometer für Auswertung

abw_wl = ((wl_mittel*10**6) - 635)/635
print('Abweichung Wellenlänge', abw_wl)

#Vakuum und Luft
print('Vakuum und Luft')
delDruck, impulse = np.genfromtxt('data_druck.txt', unpack=True)

#Normalbedingungen
T_0 = 273.15 #K
p_0 = 1013.2 #mbar

b = 50 # Größe der Messzelle in Millimeter
T = 293.15 #Umgebungstemperatur in Kelvin, aus Altprotokoll

z = unp.uarray(impulse, 4) #Impulse mit Fehler von 4
print(z)

delDruck /= 750.062 #von Torr nach Bar

n = 1 + ( (z * wl_mittel * T * p_0)/(2 * b * T_0 * delDruck) )
print('brechung', n)
n_mittel = np.mean(n)
print(n_mittel)

abw_luft = (1.00028 - n_mittel)/n_mittel
print('Abweichung brechung', abw_luft)