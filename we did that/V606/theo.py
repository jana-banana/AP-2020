import numpy as np 



#chi_t = (m0 * (mb)**2 * (gj)**2 * N * J * (J+1) )/(3 * kb * T)

m0 = 1.257e-6
mb = (-1.602e-19 * 6.626e-34)/(4*np.pi * 9.109e-31) 
kb = 1.38e-23
T = 17 + 273.15 
A = 6.02214086e23

#gj = (3*J*(J+1)+ (S*(S+1)-L*(L+1)))/(2*J*(J+1))

# Dy2O3
print('Dy2O3')
J = 7.5
S = 2.5
L = 5


M = 372.9982 #molare masse g/mol
dichte = 7.8e6 #g/m^3
N = 2*(A * dichte)/(M) #1/m^3      SOLL DIE DICHTE IN M^3 ? UND KG ODER G?

gj = (3*J*(J+1)+ (S*(S+1)-L*(L+1)))/(2*J*(J+1))
print('g_j:', gj )

chi_t = (m0 * (mb)**2 * (gj)**2 * N * J * (J+1) )/(3 * kb * T)
print('chi_T:', chi_t)

U_Br_ohne, U_Br_mit, R3_ohne, R3_mit = np.genfromtxt('daten_Dy2O3.txt', unpack=True)

R3_mit *=5 #m Ohm
R3_ohne *=5 #m Ohm
print('DIFF R', (R3_ohne - R3_mit))

Q = 15.1/(17.3*7.8) #cm^2
F = 0.866 #cm^2

print('Dy2O3')
#Suszep. mit Widerstand
print('Suszep. mit Widerstand')
x_Widerstand=[]
for i in range(3):
    x_Widerstand.append(2*(F/Q)*(R3_ohne[i] - R3_mit[i]) /1000000 )
    print(x_Widerstand[i])
print('MITTELWERT',np.mean(x_Widerstand))
print('FEHLER', np.std(x_Widerstand)/np.sqrt(3))

#Suszep. mit Brückenspannung
print('Suszep. mit Brückenspannung')
x_Spannung=[]
for i in range(3):
    x_Spannung.append(4* (F/Q)* (U_Br_mit[i] - U_Br_ohne[i])/1000)
    print(x_Spannung[i])
print('MITTELWERT',np.mean(x_Spannung))
print('FEHLER', np.std(x_Spannung)/np.sqrt(3))

abw_wid = (chi_t - np.mean(x_Widerstand)) / chi_t
print('ABWEICHUNG WiDERSTAND', abw_wid)
abw_u = (chi_t - np.mean(x_Spannung)) / chi_t
print('ABWEICHUNG SPANNUNG', abw_u)

#Gd2O3
print('Gd2O3')
J = 3.5
S = 3.5
L = 0

M = 362.4982 #molare masse g/mol
dichte = 7.4e6 #g/m^3
N = 2*(A * dichte)/(M) #1/m^3 

gj = (3*J*(J+1)+ (S*(S+1)-L*(L+1)))/(2*J*(J+1))
print('g_j:', gj )

chi_t = (m0 * (mb)**2 * (gj)**2 * N * J * (J+1) )/(3 * kb * T)
print('chi_T:', chi_t)


U_Br_ohne, U_Br_mit, R3_ohne, R3_mit = np.genfromtxt('daten_Gd2O3.txt', unpack=True)

R3_mit *=5 #m Ohm
R3_ohne *=5 #m Ohm
print('DIFF R', (R3_ohne - R3_mit))

print('Gd2O3')
Q = 14.08/(7.4*17.5)
print('Suszep. mit Widerstand')
x_Widerstand=[]
for i in range(3):
    x_Widerstand.append(2* (R3_ohne[i] - R3_mit[i]) /1000000 *(F/Q) )
    print(x_Widerstand[i])
print('MITTELWERT',np.mean(x_Widerstand))
print('FEHLER', np.std(x_Widerstand)/np.sqrt(3))

#Suszep. mit Brückenspannung
print('Suszep. mit Brückenspannung')
x_Spannung=[]
for i in range(3):
    x_Spannung.append(4* (F/Q)* (U_Br_mit[i] - U_Br_ohne[i])/1000)
    print(x_Spannung[i])
print('MITTELWERT',np.mean(x_Spannung))
print('FEHLER', np.std(x_Spannung)/np.sqrt(3))

abw_wid = (chi_t - np.mean(x_Widerstand)) / chi_t
print('ABWEICHUNG WiDERSTAND', abw_wid)
abw_u = (chi_t - np.mean(x_Spannung)) / chi_t
print('ABWEICHUNG SPANNUNG', abw_u)