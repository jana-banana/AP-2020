import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#Aufgabe a Zählrohr-Charakteristik

#Diagramm mit Fehlerbalken
U, N = np.genfromtxt('Kennlinie.txt', unpack=True)
N_err = np.sqrt(N)
plt.errorbar(U, N ,yerr=N_err, fmt ='.')
plt.xlabel('Spannung U [V]')
plt.ylabel('registrierte Teilchenanzahl N [Imp/60s]')
plt.plot(U,N, '.',label='Charakteristik')
plt.tight_layout()
plt.savefig('Zählrohrcharakteristik.pdf')

#Ausgleichsrechnung vom Plateau
U_Plateau, N_Plateau=[],[]
for i in range (23):
    U_Plateau.append(U[i+8])
    N_Plateau.append(N[i+8])

def line(U,a,b):
    return a*U+b

popt, pcov = curve_fit(line, U_Plateau, N_Plateau)

print("a =", popt[0])
print("b =", popt[1])

plt.plot(U, line(U, popt[0], popt[1]), 'r-')
plt.tight_layout()
plt.savefig('Plateau_Gerade.pdf')


print('x1 =', line(0,popt[0],popt[1]))
print('x2 =', line(100,popt[0],popt[1]))

#Aufgabe b Primär- und Nachentladungsimpulsen
