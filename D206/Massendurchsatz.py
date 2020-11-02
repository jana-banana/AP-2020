#Aufgabe e
#von güteziffer dQ_2/dt = L dm/dt

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


Z,p1,K1,K2,p2= [], [], [],[],[]
for line in open('letgo.txt', 'r'):
  values = [float(s) for s in line.split()]
  Z.append(values[0])
  K2.append(values[3]+273.15)
  K1.append(values[1]+273.15)
  p1.append(values[2]+1)
  p2.append(values[4]+1)


plt.xlabel('1/T [1/K]')
plt.ylabel("log(p/p_0)")

a,b =[],[]
for i in range(36):
  a.append(1/K1[i])
  b.append(np.log(p1[i]/p1[0]))

c,d =[],[]
for i in range(36):
  c.append(1/K2[i])
  d.append(np.log(p2[i]/p2[0]))


plt.plot(a, b, 'o',label='Reservoir 1')
plt.plot(c, d, 'o',label='Reservoir 2')


plt.tight_layout()


def line(x,m,b):
    return m*x + b

popt, pcov = curve_fit(line, a, b)

print("Reservoir 1")
print("m =", popt[0],"+/-", pcov[0,0]**0.5)
print("b =", popt[1],"+/-", pcov[1,1]**0.5)

z = np.array([Z])
plt.plot(z, popt[0]*z + popt[1], label='Ausgleichsgerade zu Reservoir 1')

popt, pcov = curve_fit(line, c, d)

print("Reservoir 2")
print("m =", popt[0],"+/-", pcov[0,0]**0.5)
print("b =", popt[1],"+/-", pcov[1,1]**0.5)

m = -popt[0]
dm = pcov[0,0]**0.5 #feheler m

plt.plot(z, popt[0]*z + popt[1], label='Ausgleichsgerade zu Reservoir 2')


print("Verdampfungswärme L")#nur für T2; [L] = J/mol
print("L = ",(m*8.314) #L=-m*R ; R= 8.314(J)/(mol*K) 

#dL/dm = R; Fehler L = R*(Fehler m)
print("Fehler von L = +/- ", 8.314*dm )

plt.legend() 
plt.savefig('build/TempDruck.pdf')