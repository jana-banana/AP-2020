import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp 

#Aufgabe a Zählrohr-Charakteristik

#Diagramm mit Fehlerbalken
U, N = np.genfromtxt('Kennlinie.txt', unpack=True)
N_err = np.sqrt(N)
plt.errorbar(U, N ,yerr=N_err, fmt ='.')
plt.xlabel('Spannung U [V]')
plt.ylabel('registrierte Teilchenanzahl N [Imp/60s]')
plt.plot(U,N, '.',label='Messpunkte' , 'b')
plt.tight_layout()
plt.savefig('Zählrohrcharakteristik.pdf')

plt.clf()

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

plt.errorbar(U, N ,yerr=N_err, fmt ='.')
plt.xlabel('Spannung U [V]')
plt.ylabel('registrierte Teilchenanzahl N [Imp/60s]')
plt.plot(U,N, '.',label='Messpunkte', 'b')
plt.tight_layout()
#plt.savefig('Zählrohrcharakteristik.pdf')


plt.plot(U, line(U, popt[0], popt[1]), label='Ausgleichgerade des Plateau-Bereichs')
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
#können wir net machen

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
    zahl = I_Z[i]/(N[4+(5*i)])
    Z.append(zahl)
    print('Strom I:', I_Z[i], '; ',zahl, '1/e')
    x = (1/(N[4+(5*i)]))**2 *(I_fehler)**2 + (I_Z[i]/(N[4+(5*i)])**2 ) * (N[4+(5*i)])
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

#Aufgabe d freigesetzte Ladung pro einfallendem Teilchen
print('Aufgabe d freigesetzte Ladung pro einfallendem Teilchen')
d_t = 60 #sekunden

print('n von den ladungen')
N_ladung = []
for i in range(8):
    N_ladung.append(N[4+(5*i)])
    print(N_ladung[i])

print('Z= I/N  (mal 1/e)') #hat einheit: mA*60s * 1/e
Zahl = []
for i in range(8):       #eig das gleiche wie aufgabe 3 dadrüber
    Zahl.append(I_Z[i]/N_ladung[i])
    print(Zahl[i])

print('d_Q = I * d_t / Z') #hier dann einheit: mA*s/ (mA*60s * 1/e) = e/60
Q=[]
for i in range(8):
    x = (I_Z[i] * d_t )/Zahl[i]
    Q.append(x)
    print(Q[i])

#fehler von Q
print('fehler von Q')
Q_fehler =[]
for i in range (8):
    x = (d_t/Zahl[i])**2 * I_fehler + ((d_t * I_Z[i])/(Zahl[i]**2))**2 * Z_fehler_quad[i]
    Q_fehler.append(np.sqrt(x))
    print(Q_fehler[i])

N = np.array([9837, 9995, 10264, 10151, 10184, 10253, 10493, 11547])
N_err = np.sqrt(N)
N0 = unp.uarray(N, N_err)

print('Q=', N0*60)
I = np.array([0.3, 0.4, 0.7, 0.8, 1.0, 1.3, 1.4, 1.8])
I_err = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
I0 = unp.uarray(I, I_err)
Z = (I0 / N0)*60 
print('Z=', Z)


Z_f= [0.000306,0.000301,0.000295,0.000299,0.000300,0.000302,0.000296,0.000274]
Z_n=[0.0018298261665141812,  0.0024012006003001503,0.004091971940763835, 0.004728598167668211,0.0058915946582875104,0.007607529503559934,0.00800533689126084, 0.009353078721745909]

plt.errorbar(I, Z_n ,yerr=Z.f, fmt ='.')
plt.xlabel('Zählerstrom I [ $\mu$A]')
plt.ylabel('Z [$\mu$As/e]')
plt.plot(I,Z_n, '.')
plt.tight_layout()
plt.savefig('Aufgabe_Bestimmung_des_Zaehlrohrstroms.pdf')