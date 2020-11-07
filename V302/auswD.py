import numpy as np 
import matplotlib.pyplot as plt 

omg_0 = (1000*409.425*10**(-9))**(-1)
nu_0 = omg_0 / (2*np.pi)
print('nu_0 = ', nu_0)

nu, U_S, U_Br = np.genfromtxt('data.txt', unpack=True)

Omg = nu/nu_0
y = U_Br / U_S

print('nu ', nu)
print('omega ', Omg)
print('y', y)
#Omg, y = [],[]
#for line in open('data.txt', 'r'):
#  values = [float(s) for s in line.split()]
#  Omg.append(values[0]/nu_0)
#  y.append(values[2]/values[1])

n = np.linspace(20, 30001, 1000)
Omg_plot = n/nu_0

def a(x):
    z = ((x**2 - 1)**2)/(9*((1 - x**2)**2 + 9* x**2))
    return z**0.5




plt.plot(Omg, y, 'k.', label='Messdaten' )
plt.plot(Omg_plot , a(Omg_plot) , label=f'Theoriekurve' )

plt.xlim(0.05, 100)
plt.ylim(0, 0.40)
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$\frac{U_Br}{U_S}$')
plt.xscale('log')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('plot.pdf')