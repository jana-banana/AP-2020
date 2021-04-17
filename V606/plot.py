import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

f, U_A = np.genfromtxt('data_selektiv.txt', unpack=True)
plt.plot(f, U_A/1000,'.', label ='Messwerte')
plt.xlabel(r'Frequenz $\nu$')
plt.ylabel(r'Spannungsverhältnis $\frac{U_A}{U_B}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('content/filterkurvePLOT.pdf')

U_Br_ohne, U_Br_mit, R3_ohne, R3_mit = np.genfromtxt('daten_Dy2O3.txt', unpack=True)

R3_mit *=5 #m Ohm
R3_ohne *=5 #m Ohm

#Dy2O3
Q = 15.1/(17.3*7.8) #cm^2
F = 0.866 #cm^2

print('Dy2O3')
#Suszep. mit Widerstand
print('Suszep. mit Widerstand')
x_Widerstand=[]
for i in range(3):
    x_Widerstand.append(2* (R3_ohne[i] - R3_mit[i]) /R3_ohne[i] *(F/Q) )
    print(x_Widerstand[i])
print('MITTELWERT',np.mean(x_Widerstand))
print('STANDARDABWEICHUNG', np.std(x_Widerstand))

#Suszep. mit Brückenspannung
print('Suszep. mit Brückenspannung')
x_Spannung=[]
for i in range(3):
    x_Spannung.append(4* (F/Q)* (U_Br_mit[i] - U_Br_ohne[i])/1000)
    print(x_Spannung[i])
print('MITTELWERT',np.mean(x_Spannung))
print('STANDARDABWEICHUNG', np.std(x_Spannung))

#Gd2O3
U_Br_ohne, U_Br_mit, R3_ohne, R3_mit = np.genfromtxt('daten_Gd2O3.txt', unpack=True)

R3_mit *=5 #m Ohm
R3_ohne *=5 #m Ohm

print('Gd2O3')
Q = 14.08/(7.4*17.5)
print('Suszep. mit Widerstand')
x_Widerstand=[]
for i in range(3):
    x_Widerstand.append(2* (R3_ohne[i] - R3_mit[i]) /R3_ohne[i] *(F/Q) )
    print(x_Widerstand[i])
print('MITTELWERT',np.mean(x_Widerstand))
print('STANDARDABWEICHUNG', np.std(x_Widerstand))

#Suszep. mit Brückenspannung
print('Suszep. mit Brückenspannung')
x_Spannung=[]
for i in range(3):
    x_Spannung.append(4* (F/Q)* (U_Br_mit[i] - U_Br_ohne[i])/1000)
    print(x_Spannung[i])
print('MITTELWERT',np.mean(x_Spannung))
print('STANDARDABWEICHUNG', np.std(x_Spannung))
