import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

X, Y, Z= [], [], []
for line in open('data.txt', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])
  Y.append(values[1]+273.15)
  Z.append(values[3]+273.15)

plt.xlim(0, 35)
plt.ylim(273.15, 340)
plt.xlabel('t [min]')
plt.ylabel('T [K]')

plt.plot(X, Y, 'o',label='T1')
plt.plot(X,Z, 'o',label ='T2')


plt.tight_layout()



def line(X, A, B, C):
    return A * X**2 + B*X + C

popt, pcov = curve_fit(line, X, Y)

print("T1")
print("A =", popt[0])
print("B =", popt[1])
print("C =", popt[2])

a = popt[0]
b = popt[1]
c = popt[2]

fkt=[]
for i in range(36):
  fkt.append(a*X[i]**2 + b*X[i]+ c)

plt.plot(X, fkt , label='Ausgleichsgerade zu T1')


popt, pcov = curve_fit(line, X, Z)

print("T2")
print("A =", popt[0],"+/-", pcov[0,0]**0.5)
print("B =", popt[1],"+/-", pcov[1,1]**0.5)
print("C =", popt[2],"+/-", pcov[2,2]**0.5)

a = popt[0]
b = popt[1]
c = popt[2]

lol=[]
for i in range(36):
  lol.append(a*X[i]**2 + b*X[i]+ c)

plt.plot(X,lol, label='Ausgleichsgerade zu T2')

plt.legend()
plt.savefig('build/tempK.pdf')