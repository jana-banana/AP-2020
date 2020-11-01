#Bestimmung der realen Güteziffer
# dQ1/dt = (m_1*c_w +m_k*c_k ) dT_1/dt
# m_1*c_w Wärmekapazität des Wasser in Reservoir 1
# m_k * c_k Wärmekapazität der Kupferschlange und des Eimers

#Für Güteziffer v =dQ1/(dt * N)
#N Leistungsaufnahme des Kompressors

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#Aufgabe b
X, Y, Z, N= [],[], [], []
for line in open('data.txt', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])
  Y.append(values[1]+273.15)
  Z.append(values[3]+273.15)
  N.append(values[5])

def line(X, A, B, C):
    return A * X**2 + B*X + C

def ableitung(X,A,B): #für Aufgabe c
    return 2*A*X + B

def realGute(ab, N): #für Aufgabe d
    return ((4*4184 + 750)*ab)/N  #J/KW

popt, pcov = curve_fit(line, X, Y)

print("T1")
print("A =", popt[0])
print("B =", popt[1])
print("C =", popt[2])

for i in range(4): #einfach die ersten vier Punkte
    print("ableitung bei t = ", X[i])
    abl = ableitung(X[i], popt[0], popt[1])
    print(abl)
    print("reale Güteziffer hier ", realGute(abl, N[i]))

popt, pcov = curve_fit(line, X, Z)

print("T2")
print("A =", popt[0])
print("B =", popt[1])
print("C =", popt[2])


for i in range(4):
    print("ableitung bei t = ", X[i])
    abl = ableitung(X[i], popt[0], popt[1])
    print(abl)
    print("reale Güteziffer hier ", realGute(abl, N[i]))




