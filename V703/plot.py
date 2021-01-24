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
plt.plot(U,N, '.',label='Messpunkte')
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
errors = np.sqrt(np.diag(pcov))
print("a =", popt[0])
print("b =", popt[1])
print("a fehler =",errors[0])
print("b fehler=",errors[1])

plt.plot(U, line(U, popt[0], popt[1]), 'r-', label='Ausgleichgerade des Plateau-Bereichs')
plt.tight_layout()
plt.legend()
plt.savefig('Plateau_Gerade.pdf')

f1=line(0,popt[0],popt[1])
f2 = line(100,popt[0],popt[1])
print('x1 = ',0, 'f1 = ', f1)
print('x2 =',100, 'f2 = ', f2)

print('Steigung auf 100V:', f2/f1)

plt.clf()
#Aufgabe b Primär- und Nachentladungsimpulsen

#Aufgabe c Totzeit

#Oszilloskop, durch Ablesen.
#Bin mir nnicht ganz sicher, aber so um 100 mikro sekunden

#Zwei-Quellen-Methode

N1= 96041/120  #impulse/s
N2= 76518/120
N1_2= 158479/120

T= (N1 + N2 - N1_2)/(2 *N1 *N2)

print('Totzeit T mit Zwei-Quellen-Methode:', T)

T_fehler_quad= ((N1_2 - N2)/(2*(N1**2)*N2))**2 *N1 + ((N1_2 - N1)/(2*(N2**2)*N1))**2 *N2 + (-1/(2*N1*N2))**2 * N1_2 
print('davon der Fehler mit Gauß:', np.sqrt(T_fehler_quad))

#Aufgabe 3. Bestimmung des Zählrohrstroms auf DatenUndHinweise

U_Z, I_Z = np.genfromtxt('Zaehlrohrstrom.dat', unpack=True)

e = 1.602 * 10**(-19) #As
I_fehler= 0.05 *10**(-6)

print( 'Zahl Z der freigesetzten Ladungen pro eingefallenen Teilchen')
Z, Z_fehler_quad, Z_fehler= [], [], []
for i in range(8):
    zahl = I_Z[i]/(N[3+(5*i)])
    Z.append(zahl)
    print('Strom I:', I_Z[i], '; ',zahl, '1/e')
    x = (1/(N[3+(5*i)]))**2 *(I_fehler)**2 + (I_Z[i]/(N[3+(5*i)])**2 ) * (N[3+(5*i)])
    Z_fehler_quad.append(x)
    
print('Fehler von Z^2')
for i in range(8):
    print(Z_fehler_quad[i])

#fehler mit gauß
print('Fehler von Z:')
for i in range(8):
    x = np.sqrt(Z_fehler_quad[i])
    Z_fehler.append(x)
    print(Z_fehler[i])

#plt.errorbar(I_Z, Z ,yerr=Z_fehler, fmt ='.')
plt.xlabel('Zählerstrom I [ $\mu$A]')
plt.ylabel('Zahl Z der freigesetzten Ladungen pro eingefallenen Teilchen')
plt.plot(I_Z,Z, '.')
plt.tight_layout()
plt.savefig('Aufgabe_Bestimmung_des_Zaehlrohrstroms.pdf')

