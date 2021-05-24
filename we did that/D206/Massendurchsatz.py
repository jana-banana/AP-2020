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
L = m*8.314

print("L = ",(m*8.314))#L=-m*R ; R= 8.314(J)/(mol*K) 

#dL/dm = R; Fehler L = R*(Fehler m)
print("Fehler von L = +/- ", 8.314*dm )
dL = 8.314*dm

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
    return ((4*4184 + 750)*ab)/N  #dimensionslos

def fehlerMassendurchsatz(X,A,B,dA,dB,L,dL):
    return ( (((4*4184 + 750)/L)*2*L*dA)**2 + (((4*4184 + 750)/L)*dB)**2 +  ( (( (4*4184 + 750)*(2*A*X+B) ) / (L**2) ) * dL )**2 )**0.5
    
popt, pcov = curve_fit(line, X, T2)

print("T2")
print("A =", popt[0], "+/-", pcov[0,0]**0.5)
print("B =", popt[1], "+/-", pcov[1,1]**0.5)
print("C =", popt[2], "+/-", pcov[2,2]**0.5)

A = popt[0]
B = popt[1] 
dA = pcov[0,0]**0.5
dB = pcov[1,1]**0.5


def kompress(p_a,p_b,MaDu): #Aufgabe f Bestimmung der mechanischen Kompressorleistung N_mech
  k=1.4
  p0 = p2[0]*100000
  roh = 5.51
  T0 = T2[0]
  return (1/(k-1)) * (p_b * (p_a/p_b)**(1/k) - p_a ) * ((T*p0)/(roh*T0*p_a)) * MaDu


for i in range(1,5):
  print()
  print("ableitung bei t [s]= ", X[i]*60)
  abl = ableitung(X[i], popt[0], popt[1])
  MaDu = (realGute(abl, N[i])) / L
  print("Massendurchsatz hier: ", MaDu)
  print("Fehler Massendurchstz: ",fehlerMassendurchsatz(X[i],A,B,dA,dB,L,dL))
  print()
  print("Bestimmung der mechanischen Kompressorleistung N_mech")
  print("mech Kompr.Leistung hier ", kompress(p1[i]*100000,p2[i]*100000,T2[i],MaDu))

plt.legend() 
plt.savefig('build/TempDruck.pdf')