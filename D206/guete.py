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
X, T1, T2, N= [],[], [], []
for line in open('data.txt', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])
  T1.append(values[1]+273.15)
  T2.append(values[3]+273.15)
  N.append(values[5])

def line(X, A, B, C):
    return A * X**2 + B*X + C

def ableitung(X,A,B): #für Aufgabe c
    return 2*A*X + B

def realGute(ab, N): #für Aufgabe d
    return ((4*4184 + 750)*ab)/N  #J/KW

def idealGute(a,b):
    return a/(a-b)

popt, pcov = curve_fit(line, X, T1)

print("T1")
print("A =", popt[0], "+/-", pcov[0,0]**0.5)
print("B =", popt[1], "+/-", pcov[1,1]**0.5)
print("C =", popt[2], "+/-", pcov[2,2]**0.5)

for i in range(1,5): #einfach die ersten vier Punkte
    print("ableitung bei t = ", X[i])
    abl = ableitung(X[i], popt[0], popt[1])
    print(abl)
    print("reale Güteziffer hier ", realGute(abl, N[i]))
    print("ideale Güteziffer hier ", idealGute(T1[i], T2[i]))

popt, pcov = curve_fit(line, X, T2)

print("T2")
print("A =", popt[0], "+/-", pcov[0,0]**0.5)
print("B =", popt[1], "+/-", pcov[1,1]**0.5)
print("C =", popt[2], "+/-", pcov[2,2]**0.5)



for i in range(1,5):
    print("ableitung bei t = ", X[i])
    abl = ableitung(X[i], popt[0], popt[1])
    print(abl)
    print("reale Güteziffer hier ", realGute(abl, N[i]))
    print("ideale Güteziffer hier ", idealGute(T1[i], T2[i]))





