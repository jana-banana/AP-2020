import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

import os

if os.path.exists("build") == False:
    os.mkdir("build")

###allgemeine Definitionen
d = 201.4e-12 #meter
Ea_lit = 8048.1 #eV
Eb_lit = 8906.9 #eV
tau = 90e-6 #sec
lamc_lit = const.h /( const.m_e * const.c ) 


def E(t): #ohne Umrechnung in Joule, aktuell in eV
    z = const.h * const.c
    n = 2 * d * unp.sin(t * np.pi / 180) * const.e
    return z/n

def diff(exp, lit): 
    d = 1 - (exp/lit)
    return d * 100 #prozent

def lam(theta): 
    l = 2 * d * unp.sin(theta * np.pi / 180)
    return l

def tot(N):
    tau = 90e-6
    n = 1 - tau*N
    return N/n
#---------------------------------------------------------------------------------------------------------------------------------------------------------
### 1.Analyse des Emissionsspektrum
print('1.Analyse des Emissionsspektrum')

t, N = np.genfromtxt('data/emissionsspektrum.txt', unpack=True)

#sind die bezeichnungen a und b richtig, jetzt schon 
t_a = ufloat(t[145], 0.1)
N_a = N[145]
print('k_a', t_a, N_a)

t_b = ufloat(t[122], 0.1)
N_b = N[122]
print('k_b', t_b, N_b)

print('lambda_a', lam(t_a))
print('lambda_b', lam(t_b))

print('E_a', E(t_a))
print('E_b', E(t_b)) 

#Abweichungen
print( 'diff E_alpha', diff(E(t_a), Ea_lit))
print( 'diff E_beta', diff(E(t_b), Eb_lit))


#plot
plt.figure()
plt.plot(t, N, 'k.', label='Messwerte Emissionsspektrum')
plt.xlabel(r'$ \alpha_{\text{GM}} \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.scatter([11.1], [420.0], s=20, marker='x', color='red')
plt.annotate(r'Bremsberg', 
            xy = (11.1, 420.0), xycoords='data', xytext=(-10, 20),
            textcoords='offset points', fontsize=12, 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))
plt.scatter([t_b.n], [1599.0], s=20, marker='x', color='red')
plt.annotate(r'$K_{\beta}$',
            xy = (t_b.n, 1599.0), xycoords='data', xytext=(-50, -25),
            textcoords='offset points', fontsize=12,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))
plt.scatter([t_a.n], [5050.0], s=20, marker='x', color='red')
plt.annotate(r'$K_{\alpha}$',
            xy = (t_a.n, 5050.0), xycoords='data', xytext=(+10, -2),
            textcoords='offset points', fontsize=12,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))
plt.ylim(0,5200)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/emissionsspektrum.pdf')


#----------------------------------------------------------------------------------------------------------------
###2. Bestimmung der Transmission als Funktion der Wellenlänge

t_al , N_al = np.genfromtxt('data/compton_mit.txt', unpack=True)
t_0  , N_0  = np.genfromtxt('data/compton_ohne.txt', unpack=True)

lamd = 2 * d * np.sin(t_0 * np.pi / 180)
# lamd = lam(t_0)

#Poisson-Fehler auf Anzahl setzten 
t = 200 #second
N0_err = np.sqrt(N_0 /t)
Nal_err =np.sqrt(N_al /t)

#let's make some ufloats
N0 = unp.uarray(N_0, N0_err)
Nal = unp.uarray(N_al, Nal_err)

#Totzeitkorrektur
I_0 = tot(N0)
I_al = tot(Nal)

#Transmission
T = I_al / I_0

#ausgleichsgerade
params, cov = np.polyfit(lamd , unp.nominal_values(T), deg=1, cov=True)
errs = np.sqrt(np.diag(cov))
for name, value, error in zip('ab', params, errs):
    print(f'{name} = {value:.3f} +- {error:.3f}')

#plot 
lamd_start = lam(7)
lamd_stop = lam(10) 
lamd_plot = np.linspace(lamd_start, lamd_stop, 1000)

plt.figure()
plt.plot(lamd ,unp.nominal_values(T), 'r.', label="Messwerte")
plt.plot(lamd_plot, params[0]*lamd_plot + params[1], 'k-', label='Lineare Regression')
plt.errorbar(lamd , unp.nominal_values(T), yerr=unp.std_devs(T), fmt='r_')
plt.grid()
plt.legend()
plt.xlabel(r'Wellenlänge $\lambda \, \mathbin{/} \si{\metre}$')
plt.ylabel(r'Transmission')
plt.savefig('build/transmission.pdf')

#---------------------------------------------------------------------------------------------------------
#3.Bestimmung der Compton-Wellenlänge

#Daten einlesen
I_0 = 2731 #Impulse
I_1 = 1180 #Impulse
I_2 = 1024 #Impulse

#Transmissionen
T_1 = I_1 / I_0
T_2 = I_2 / I_0
print('T_1:', T_1)
print('T_2:', T_2)

a = unp.uarray(params[0], errs[0])
b = unp.uarray(params[1], errs[1])

#wellenlängen ausrechnen
lamd_1 = (T_1 - b)/a
lamd_2 = (T_2 - b)/a
print('lambda_1', lamd_1)
print('lambda_2', lamd_2)

lamd_c = lamd_2 - lamd_1
print('Compton-Wellenlänge:', lamd_c)
print('diff', diff(lamd_c, lamc_lit))