import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
############################# TEIL 1 ################################
d_y, d_x = np.genfromtxt('data/Steigungsdreiecke_schwarz_24_grad.txt', unpack=True)
d_x /= 20 #volt
d_y *= (0.01) #muA

print('d_U_A, ', d_x)
print('d_I_A, ', d_y)

U_A, I_A = [], []
I_A.append(1.7) #muA, bei U_A=0 
U_A.append(0)
for i in range(21):
    I_A.append(I_A[i] - d_y[i])
    U_A.append(U_A[i] + d_x[i])

print('I_A, ', I_A)
print('U_A, ', U_A)


plt.plot(U_A, I_A,'b.', label='schwarze Kurve')

plt.xlabel('$U_A  $ / V')
plt.ylabel('$I_A  / \mu $ A')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('bilder/integral.pdf')
plt.clf()

plt.plot(U_A, -1*d_y,'b.', label='schwarze Kurve abstand')
plt.xlabel('$U_A  $ / V')
plt.ylabel('$ I_A / \mu $ A')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('bilder/abstande.pdf')
plt.clf()

steigung = d_y/d_x

plt.plot(U_A, steigung,'b.', label='schwarze Kurve ableitung')
plt.xlabel('$U_A  $ / V')
plt.ylabel('$ I_A/U_A / \mu $A/V')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('bilder/ableitung.pdf')
plt.clf()
################################### TEIL 2 ######################################
d_x_200 = np.genfromtxt('data/abstand_200.txt', unpack=True)
d_x_176 = np.genfromtxt('data/abstand_176.txt', unpack=True)

d_x_176 *= (40/208)
d_x_200 *= (40/208)

print('d_x_200: ', d_x_200)
print('d_x_176: ', d_x_176)

print('mittel d_x_176 ', np.mean(d_x_176))
print('mittel d_x_200 ', np.mean(d_x_200))

h = 6.62607015e-34 #Js
c = 299792458
d_x_176_mittel= np.mean(d_x_176)
d_x_200_mittel= np.mean(d_x_200)

lambda_176 = (h*c)/(d_x_176_mittel * 1.60218e-19)
lambda_200 = (h*c)/(d_x_200_mittel *1.60218e-19)
ap = (h*c)/(5.22 * 1.60218e-19)

print('lamda_176 ', lambda_176)
print('lamda_200 ', lambda_200)
print('AP ', ap)