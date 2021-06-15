import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

#AUFGABE A
print('AUFGABE A')

t, U_c= np.genfromtxt('data/data_a.txt', unpack=True)


def line_a(t,RC,b):
    #return a*t + b
    return b *np.exp(-1 * t /RC)

popt, pcov = curve_fit(line_a, t, U_c)
errors = np.sqrt(np.diag(pcov))
print("RC =", popt[0])
print("b =", popt[1])
print("RC fehler =",errors[0])
print("b fehler=",errors[1])

plt.plot(t , U_c , '.', label = 'Messwerte')
plt.yscale('log')
plt.xlabel('Zeit t [s]')           #EINHEIT
plt.ylabel('Kondensatorspannung $U_c$ [V]') #EINHEIT

plt.plot(t, line_a(t, popt[0], popt[1]),'r-', label='Ausgleichgerade')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('content/data_a_ausgleich.pdf')

plt.clf()

#AUFGABE B
print('AUFGABE B')

f, U_a= np.genfromtxt('data/data_b.txt', unpack=True)

U_0 = 6.2 # aus messung. keine ahnung was U_0 ist.
A = U_a / U_0

def line_b(f, RC):
    return 1/np.sqrt(1 + (2* np.pi* f)**2 * RC**2)

popt, pcov = curve_fit(line_b, f, A)
errors = np.sqrt(np.diag(pcov))
print("RC =", popt[0])
print("RC fehler =",errors[0])

plt.plot(f , A , '.', label = 'Messwerte')
plt.xscale('log')
plt.xlabel('Frequenz f [Hz]')        #EINHEIT
plt.ylabel('$U_a / U_0$')          #EINHEIT

plt.plot(f, line_b(f, popt[0]),'r-', label='Ausgleichgerade')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('content/data_b_ausgleich.pdf')

plt.clf()
#AUFGABE C
print('AUFGABE C')

f, a, b= np.genfromtxt('data/data_c.txt', unpack=True)
phi = (a/b) * 2* np.pi
print('PHASENVERSCHIEBUNG  ', phi)

def line_c(f, RC):
    return np.arctan(-2* np.pi* f* RC)

plt.plot(f , phi , '.', label = 'Messwerte')
plt.xscale('log')
plt.xlabel('Frequenz f [Hz]')        #EINHEIT
plt.ylabel('Phasenverschiebung φ')          #EINHEIT

popt, pcov = curve_fit(line_c, f, phi)
errors = np.sqrt(np.diag(pcov))
print("RC =", popt[0])
print("RC fehler =",errors[0])

RC = popt[0]

plt.plot(f, line_c(f, popt[0]),'r-', label='Ausgleichgerade')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('content/data_c_ausgleich.pdf')

plt.clf()
#AUFGABE D
print('AUFGABE D')

#y = np.sin(phi)/(2* np.pi* f* RC)
#r = np.sqrt( (2* np.pi* f)**2 + (y)**2 )
#alpha = np.arctan(y/f)
#plt.axes(projection = 'polar')
#rads = np.arange(0, (2*np.pi), 0.01)
#plt.polar(r, alpha)
#plt.savefig('content/data_d_ausgleich.pdf')

U0 = 6.2 #Volt. Einmalige Messung

def AU(fr, ph, RC):
    -np.sin(ph)/(2*np.pi*fr*RC)

AK = 1/ (np.sqrt(1+((2*np.pi*f)**2)*(RC**2)))

plt.figure()
plt.polar(phi, AK, '.', label = 'Messwerte')

x = np.linspace(0, 100000, 10000)
phi = np.arcsin(((x*-RC)/(np.sqrt(1+x**2*(-RC)**2)))) #-RC weil wert negativ

A = 1/(np.sqrt(1+x**2*(-RC)**2))
plt.polar(phi, A, label = 'Berechnete Amplitude')
plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/data_d_ausgleich.pdf')