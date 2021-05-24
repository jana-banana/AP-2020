import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import os

if os.path.exists("build") == False:
    os.mkdir("build")

#curve-fit
#params_vi , ma_vi = np.polyfit(U_vi,np.sqrt(I_vi), deg =1, cov = True)
#errors_vi = np.sqrt(np.diag(ma_vi))


x_plot = np.linspace(-6, 5, 1000)
x_ge = np.linspace(-20,20,10000)
#rot
U_rot, Irot = np.genfromtxt('data_rot.txt', unpack=True)
I_rot = np.sqrt(Irot)
#grün
U_gr1, I1 = np.genfromtxt('data_grun1.txt', unpack=True)
I_gr1 = np.sqrt(I1)
U_gr2, I2 = np.genfromtxt('data_grun2.txt', unpack=True)
I_gr2 = np.sqrt(I2)
#violett
U_vi, Ivi = np.genfromtxt('data_violett.txt', unpack=True)
I_vi = np.sqrt(Ivi)
#gelb
U_ge, Ige = np.genfromtxt('data_gelb.txt', unpack=True)
I_ge = np.sqrt(Ige)


###curve fit und Nullstelle
#rot
params , ma = np.polyfit(U_rot[0:27],I_rot[0:27], deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a_rot = ufloat(params[0], errors[0])
c_rot = ufloat(params[1], errors[1])
print('rot')
print('a', a_rot)
print('c', c_rot)

N_rot = - c_rot / a_rot
print('N', N_rot)

#grün
params , ma = np.polyfit(U_gr1[2:17] ,I_gr1[2:17], deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a_gr1 = ufloat(params[0], errors[0])
c_gr1 = ufloat(params[1], errors[1])
print('grün')
print('a1', a_gr1)
print('c1', c_gr1)

N_gr1 = - c_gr1 / a_gr1
print('N', N_gr1)

params , ma = np.polyfit(U_gr2[0:8] ,I_gr2[0:8], deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a_gr2 = ufloat(params[0], errors[0])
c_gr2 = ufloat(params[1], errors[1])
print('a2', a_gr2)
print('c2', c_gr2)

N_gr2 = - c_gr2 / a_gr2
print('N', N_gr2)

N_gr = (N_gr1 + N_gr2)/2
print('N_gr', N_gr)

#violett
params , ma = np.polyfit(U_vi[0:13],I_vi[0:13], deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a_vi = ufloat(params[0], errors[0])
c_vi = ufloat(params[1], errors[1])
print('violett')
print('a', a_vi)
print('c', c_vi)

N_vi = - c_vi / a_vi
print('N', N_vi)

#gelb
params , ma = np.polyfit(U_ge[11:42],I_ge[11:42], deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a_ge = ufloat(params[0], errors[0])
c_ge = ufloat(params[1], errors[1])
print('gelb')
print('a', a_ge)
print('c', c_ge)

N_ge = - c_ge / a_ge
print('N', N_ge)



###Abbildungen

#rot
plt.figure()
plt.plot(U_rot, I_rot, 'k.', label='Messwerte')
plt.plot(x_plot, a_rot.n*x_plot + c_rot.n, '-r', label='Ausgleichsrechnung')
plt.xlim(-6,6)
plt.ylim(-1, 16)
plt.xlabel(r'$ U_{\text{Br}} \quad [\si{\volt}]$')
plt.ylabel(r'$ (I)^{\frac{1}{2}} \quad [(\si{\pico\ampere})^{\frac{1}{2}}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/rot.pdf')

#grün
plt.figure()
plt.plot(U_gr1, I_gr1, 'k.', label='Messwerte 1')
plt.plot(U_gr2, I_gr2, 'b.', label='Messwerte 2')
plt.plot(x_plot, a_gr1.n*x_plot + c_gr1.n, '-g', label='Ausgleichsrechnung für die Messwerte 1')
plt.plot(x_plot, a_gr2.n*x_plot + c_gr2.n, '-g', label='Ausgleichsrechnung für die Messwerte 2')
plt.xlim(-6,6)
plt.ylim(-1, 120)
plt.xlabel(r'$ U_{\text{Br}} \quad [\si{\volt}]$')
plt.ylabel(r'$ (I)^{\frac{1}{2}} \quad [(\si{\pico\ampere})^{\frac{1}{2}}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/grun.pdf')

#violett
plt.figure()
plt.plot(U_vi, I_vi, 'k.', label='Messwerte')
plt.plot(x_plot, a_vi.n*x_plot + c_vi.n, '-m', label='Ausgleichsrechnung')
plt.xlim(-6,6)
plt.ylim(-1, 150)
plt.xlabel(r'$ U_{\text{Br}} \quad [\si{\volt}]$')
plt.ylabel(r'$ (I)^{\frac{1}{2}} \quad [(\si{\pico\ampere})^{\frac{1}{2}}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/violett.pdf')

#gelb
plt.figure()
plt.plot(U_ge, I_ge, 'k.', label='Messwerte')
plt.plot(U_ge[11:42], I_ge[11:42], 'b.', label='für Ausgleichsrechnung benutzte Messwerte')
plt.plot(x_ge, a_ge.n*x_ge + c_ge.n, '-y', label='Ausgleichsrechnung')
plt.xlim(-19,19)
plt.ylim(-1, 100)
plt.xlabel(r'$ U_{\text{Br}} \quad [\si{\volt}]$')
plt.ylabel(r'$ (I)^{\frac{1}{2}} \quad [(\si{\pico\ampere})^{\frac{1}{2}}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/gelb_a.pdf')


####b
lambda_ro = 615*1e-9
lambda_gr = 546*1e-9
lambda_vi = 435*1e-9
lambda_ge = 578*1e-9

nu_ro = const.c / lambda_ro 
nu_gr = const.c / lambda_gr 
nu_vi = const.c / lambda_vi 
nu_ge = const.c / lambda_ge 
nu_plot = np.linspace(4.8e14, 7e14, 100000)

print('rot','lambda', lambda_ro, 'nu', nu_ro)
print('grün','lambda', lambda_gr, 'nu', nu_gr)
print('violett','lambda', lambda_vi, 'nu', nu_vi)
print('gelb','lambda', lambda_ge, 'nu', nu_ge)

nu = np.array([nu_ro, nu_gr, nu_vi, nu_ge])
U_g = np.array([N_rot.n, N_gr.n, N_vi.n, N_ge.n])

params , ma = np.polyfit(nu ,U_g, deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a = ufloat(params[0], errors[0])
c = ufloat(params[1], errors[1])
print('a', a.n, '+-', a.s)
print('c', c)

A_k = c*const.e
print('A_k', A_k, 'J', c, 'eV')
print('h/e0', const.h/const.e)
print('diff', (1 - (a*const.e/const.h)).n,'+-' ,(1 - (a*const.e/const.h)).s )

plt.figure()
plt.plot(nu_ro, N_rot.n, 'r.', label='Grenzspannung von Rot')
plt.plot(nu_gr, N_gr.n, 'g.', label='Grenzspannung von Grün')
plt.plot(nu_vi, N_vi.n, 'm.', label='Grenzspannung von Violett')
plt.plot(nu_ge, N_ge.n, 'y.', label='Grenzspannung von Gelb')
plt.plot(nu_plot, a.n*nu_plot + c.n, 'k-', label='Ausgleichsgerade')
plt.ylabel(r'$ U_{\text{G}} \quad [\text{V}]$')
plt.xlabel(r'$ \nu \quad [\text{Hz}]$')
plt.legend()
plt.savefig('build/b.pdf')


###c
plt.figure()
plt.plot(U_ge, Ige, 'y.', label='Messwerte')
plt.xlabel(r'$ U_{\text{Br}} \quad [\si{\volt}]$')
plt.ylabel(r'$ I \quad [\si{\pico\ampere}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/gelb_c.pdf')