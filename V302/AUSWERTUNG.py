import numpy as np 
import matplotlib.pyplot as plt 

#a
R_2 = np.array([332, 664, 1000])
R_3 = np.array([594, 422, 327])
R_4 = np.array([406, 578, 673])
R_x = R_2*(R_3/R_4)
print('R_x ', R_x)
print('Mittelwert ' ,np.mean(R_x))
print('Fehler des Mittelwertes ' ,np.std(R_x) / np.sqrt(len(R_x)))
#Fehler aus den Toleranzen
errR_x = np.mean(R_x)*(np.sqrt(0.002**2+0.005**2))
print('Fehler aus Toleranzen ', errR_x)
R_2 = np.array([332, 664, 1000])
R_3 = np.array([539, 368, 279])
R_4 = np.array([461, 632, 721])
R_x = R_2*(R_3/R_4)
print('R_x ', R_x)
print('Mittelwert ' ,np.mean(R_x))
print('Fehler des Mittelwertes ' ,np.std(R_x) / np.sqrt(len(R_x)))
errR_x = np.mean(R_x)*(np.sqrt(0.002**2+0.005**2))
print('Fehler aus Toleranzen ', errR_x)

#b
C_2 = np.array([597, 994])
R_2 = np.array([278, 169])
R_3 = np.array([670, 770])
R_4 = np.array([330, 230])
R_x = R_2*(R_3/R_4)
print('R_x ', R_x)
print('Mittelwert ' ,np.mean(R_x))
print('Fehler des Mittelwertes ' ,np.std(R_x) / np.sqrt(len(R_x)))
errR_x = np.mean(R_x)*(np.sqrt(0.03**2+0.005**2))
print('Fehler aus Toleranzen ', errR_x)

C_x = C_2*(R_4/R_3)
print('C_x ', C_x)
print('Mittelwert ' ,np.mean(C_x))
print('Fehler des Mittelwertes ' ,np.std(C_x) / np.sqrt(len(C_x)))
errC_x = np.mean(C_x)*(np.sqrt(0.002**2+0.005**2))
print('Fehler aus Toleranzen ', errC_x)

#c
R_2 = np.array([64, 78])
R_3 = np.array([874, 570])
R_4 = np.array([126, 430])
R_x = R_2*(R_3/R_4)
print('R_x ', R_x)
errR_x = R_x*(np.sqrt(0.03**2+0.005**2))
print('Fehler aus Toleranzen ', errR_x)
L_2 = 20.1
L_x = L_2*(R_3/R_4)
print('L_x ', L_x)
errL_x = L_x*(np.sqrt(0.002**2 + 0.005**2))
print('Fehler aus Toleranzen ', errL_x)

#e - Klirrfaktor

def a(x):
    z = ((x**2 - 1)**2)/(9*((1 - x**2)**2 + 9* x**2))
    return z**0.5
print('f(2) = ', a(2))

U_1 = 20.0 #Volt
U_Br = 0.129 #Volt

U_2 = U_Br / a(2)
print('U_2 = ', U_2)
k = U_2 / U_1
print('k = ', k)