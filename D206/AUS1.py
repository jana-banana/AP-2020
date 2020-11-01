import matplotlib.pyplot as plt
import numpy as np

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
plt.legend()
plt.title('Temperaturverlauf der Reservoire während der Messung')

plt.tight_layout()
plt.savefig('tempK.png')

