import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

import os

if os.path.exists("build") == False:
    os.mkdir("build")

#konstanten
d = 201.4e-12 #meter
E_abs = 8980.476 #eV
z = 29
R = 13.6 #eV Rydbergenergie


def E(t): #ohne Umrechnung in Joule, aktuell in eV
    z = const.h * const.c
    n = 2 * d * unp.sin(t * np.pi / 180) * const.e
    return z/n

def s(E,Z):
     w = (E)/(R) - (const.alpha**2 * Z**4)/(4)
     print(w)
     return Z - unp.sqrt(w)

def rvsline(Ik, m, n):
    return (Ik - n ) / m


### 1.Überprüfung der Bragg-Bedingung
print('Überprüfung der Bragg Bedingung')
a, N = np.genfromtxt('bragg.txt', unpack=True)

N_max =  np.amax(N)
print('a_max', a[22])
print('N_max', N[22])
print('N_max', N_max)

#Theoriewert Bragg:

a_theo = 28
dif = (a_theo - a[22])/a_theo
print('abweichung', dif)


plt.figure()
plt.plot(a, N, 'k.', label='Messwerte Bragg-Bedingung')
plt.vlines(x= 28, ymin= 50, ymax= 220 ,linewidth=1, color='r', label='Sollwinkel')
plt.scatter([28.2], [218.0], s=20, marker='x', color='blue', label='Maximum')
plt.xlabel(r'$ \alpha_{\text{GM}} \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/bragg.pdf')



### 2.Analyse des Emissionsspektrum
print('Analyse des Emissionsspektrum')

t, N = np.genfromtxt('emissionsspektrum.txt', unpack=True)

#sind die bezeichnungen a und b richtig, jetzt schon 
t_a = ufloat(t[145], 0.1)
N_a = N[145]
print('k_a', t_a, N_a)

t_b = ufloat(t[122], 0.1)
N_b = N[122]
print('k_b', t_b, N_b)


print('FWHM')
b_min = ufloat(20.155, 0.2)
b_max = ufloat(20.57, 0.2)
a_min = ufloat(22.35, 0.2)
a_max = ufloat(22.85, 0.2)
print('a_min', a_min)
print('a_max', a_max)
print('b_min', b_min)
print('b_max', b_max)

print('Auflösungsvermögen')

print('E_a', E(t_a))
print('E_b', E(t_b))
print('E_FWHMa', -E(a_max)+E(a_min))
print('E_FWHMb', -E(b_max)+E(b_min))

A_a = E(t_a)/(-E(a_max)+E(a_min))
A_b = E(t_b)/(-E(b_max)+E(b_min))

print('A_a', A_a)
print('A_b', A_b)

print('Absorptionskoeffizienten')
s1 = z - unp.sqrt(E_abs/R)
s2 = z - 2* unp.sqrt((E_abs - E(t_a))/R)
s3 = z - 3* unp.sqrt((E_abs - E(t_b))/R)
print('s1', s1)
print('s2', s2)
print('s3', s3)



plt.figure()
plt.plot(t, N, 'k.', label='Messwerte Emissionsspektrum')
plt.hlines(y= 799.5, xmin= b_min.n, xmax= b_max.n ,linewidth=1, color='b', label=r'FWHM für $K_{\beta}$')
plt.vlines(x= b_min.n, ymin= 0, ymax= 799.5 ,linewidth=1, color='b')
plt.vlines(x= b_max.n, ymin= 0, ymax= 799.5 ,linewidth=1, color='b')
plt.hlines(y= 2525, xmin=a_min.n, xmax=a_max.n ,linewidth=1, color='g', label=r'FWHM für $K_{\alpha}$')
plt.vlines(x= a_min.n , ymin= 0, ymax= 2525 ,linewidth=1, color='g')
plt.vlines(x= a_max.n , ymin= 0, ymax= 2525 ,linewidth=1, color='g')
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



###Absorptionsspektrum
print('Absorptionsspektrum')

#Zink
t_zink, N_zink = np.genfromtxt('zink.txt', unpack=True)

N_zinkmax = np.amax(N_zink)
N_zinkmin = np.amin(N_zink)
I_zink = N_zinkmin + (N_zinkmax - N_zinkmin)/2
print('Zink', 'min', N_zinkmin, 'max', N_zinkmax)
print('I_zink', I_zink)

###correction
#curve-fit
x_zink = np.array([t_zink[6],t_zink[7]])
y_zink = np.array([N_zink[6], N_zink[7]])
zinkplot = np.linspace(t_zink[6], t_zink[7], 1000)

params = np.polyfit( x_zink, y_zink, deg =1, cov = False)

m_zink = params[0]
n_zink = params[1]
print('m_zink', m_zink)
print('n_zink', n_zink)

# print(rvsline(I_zink[1], m_zink, n_zink))
tzink = ufloat(rvsline(I_zink, m_zink, n_zink), 0.05)    
Ezink = E(tzink)
Sigma_zink = s(Ezink,30)

print('t_zink', tzink)
print('E_zink', E(tzink))
print('Sigma_zink', Sigma_zink)
print('dif', 1 - 3.56/Sigma_zink.n)


#Gallium
t_gallium, N_gallium = np.genfromtxt('gallium.txt', unpack=True)

N_galliummax =  np.amax(N_gallium)
N_galliummin =  np.amin(N_gallium)
I_gallium = N_galliummin + (N_galliummax - N_galliummin)/2
print('Gallium', 'min', N_galliummin, 'max', N_galliummax)
print('I_gallium', I_gallium)

#curve-fit
x_gallium = np.array([t_gallium[3], t_gallium[4]])
y_gallium = np.array([N_gallium[3], N_gallium[4]])
galliumplot = np.linspace(t_gallium[3], t_gallium[4], 1000)

params = np.polyfit( x_gallium, y_gallium, deg =1, cov = False)

m_gallium = params[0]
n_gallium = params[1]

print('m_gallium', m_gallium)
print('n_gallium', n_gallium)

# print(rvsline(I_gallium[1], m_gallium, n_gallium))
tgallium = ufloat(rvsline(I_gallium, m_gallium, n_gallium), 0.05)
Egallium = E(tgallium)
Sigma_gallium = s(Egallium,31)

print('t_gallium', tgallium)
print('E_gallium', E(tgallium))
print('Sigma_gallium', Sigma_gallium)
print('dif', 1 - 3.62/Sigma_gallium.n)

#brom
t_brom, N_brom = np.genfromtxt('brom.txt', unpack=True)

N_brommax =  np.amax(N_brom)
N_brommin =  np.amin(N_brom)
I_brom = N_brommin + (N_brommax - N_brommin)/2
print('brom', 'min', N_brommin, 'max', N_brommax)
print('I_brom', I_brom)

#curve-fit
x_brom = np.array([t_brom[3], t_brom[5]])
y_brom = np.array([N_brom[3], N_brom[5]])
bromplot = np.linspace(t_brom[3], t_brom[5], 1000)

params = np.polyfit( x_brom, y_brom, deg =1, cov = False)

m_brom = params[0]
n_brom = params[1]

print('m_brom', m_brom)
print('n_brom', n_brom)

# print(rvsline(I_brom[1], m_brom, n_brom))
tbrom = ufloat(rvsline(I_brom, m_brom, n_brom), 0.05)
Ebrom = E(tbrom)
Sigma_brom = s(Ebrom,35)

print('t_brom', tbrom)
print('E_brom', E(tbrom))
print('Sigma_brom', Sigma_brom)
print('dif', 1 - 3.84/Sigma_brom.n)

#rubidium
t_rubidium, N_rubidium = np.genfromtxt('rubidium.txt', unpack=True)

N_rubidiummax =  np.amax(N_rubidium)
N_rubidiummin =  np.amin(N_rubidium)
I_rubidium = N_rubidiummin + (N_rubidiummax - N_rubidiummin)/2
print('rubidium', 'min', N_rubidiummin, 'max', N_rubidiummax)
print('I_rubidium', I_rubidium)

#curve-fit
x_rubidium = np.array([t_rubidium[5], t_rubidium[6]])
y_rubidium = np.array([N_rubidium[5], N_rubidium[6]])
rubidiumplot = np.linspace(t_rubidium[5], t_rubidium[6], 1000)

params = np.polyfit( x_rubidium, y_rubidium, deg =1, cov = False)

m_rubidium = params[0]
n_rubidium = params[1]

print('m_rubidium', m_rubidium)
print('n_rubidium', n_rubidium)

# print(rvsline(I_rubidium[1], m_rubidium, n_rubidium))
trubidium = ufloat(rvsline(I_rubidium, m_rubidium, n_rubidium), 0.05)
Erubidium = E(trubidium)
Sigma_rubidium = s(Erubidium,37)

print('t_rubidium', trubidium)
print('E_rubidium', E(trubidium))
print('Sigma_rubidium', Sigma_rubidium)
print('dif', 1 - 3.95/Sigma_rubidium.n)

#strontium
t_strontium, N_strontium = np.genfromtxt('strontium.txt', unpack=True)

N_strontiummax =  np.amax(N_strontium)
N_strontiummin =  np.amin(N_strontium)
I_strontium = N_strontiummin + (N_strontiummax - N_strontiummin)/2
print('strontium', 'min', N_strontiummin, 'max', N_strontiummax)
print('I_strontium', I_strontium)

#curve-fit
x_strontium = np.array([t_strontium[5], t_strontium[6]])
y_strontium = np.array([N_strontium[5], N_strontium[6]])
strontiumplot = np.linspace(t_strontium[5], t_strontium[6], 1000)

params = np.polyfit( x_strontium, y_strontium, deg =1, cov = False)

m_strontium = params[0]
n_strontium = params[1]

print('m_strontium', m_strontium)
print('n_strontium', n_strontium)

# print(rvsline(I_strontium[1], m_strontium, n_strontium))
tstrontium = ufloat(rvsline(I_strontium, m_strontium, n_strontium), 0.05)
Estrontium = E(tstrontium)
Sigma_strontium = s(Estrontium,38)
print('t_strontium', tstrontium)
print('E_strontium', E(tstrontium))
print('Sigma_strontium', Sigma_strontium)
print('dif', 1 - 3.99/Sigma_strontium.n)

#zirkonium
t_zirkonium, N_zirkonium = np.genfromtxt('zirkonium.txt', unpack=True)

N_zirkoniummax =  np.amax(N_zirkonium)
N_zirkoniummin =  np.amin(N_zirkonium)
I_zirkonium = N_zirkoniummin + (N_zirkoniummax - N_zirkoniummin)/2
print('zirkonium', 'min', N_zirkoniummin, 'max', N_zirkoniummax)
print('I_zirkonium', I_zirkonium)

#curve-fit
x_zirkonium = np.array([t_zirkonium[4], t_zirkonium[5]])
y_zirkonium = np.array([N_zirkonium[4], N_zirkonium[5]])
zirkoniumplot = np.linspace(t_zirkonium[4], t_zirkonium[5], 1000)

params = np.polyfit( x_zirkonium, y_zirkonium, deg =1, cov = False)

m_zirkonium = params[0]
n_zirkonium = params[1]

print('m_zirkonium', m_zirkonium)
print('n_zirkonium', n_zirkonium)

# print(rvsline(I_zirkonium[1], m_zirkonium, n_zirkonium))
tzirkonium = ufloat(rvsline(I_zirkonium, m_zirkonium, n_zirkonium), 0.05)
Ezirkonium = E(tzirkonium)
Sigma_zirkonium = s(Ezirkonium,40)
print('t_zirkonium', tzirkonium)
print('E_zirkonium', E(tzirkonium))
print('Sigma_zirkonium', Sigma_zirkonium)
print('dif', 1 - 4.39/Sigma_zirkonium.n)


#Bestimmung der Rydbergenergie
print('Bestimmung der Rydbergenergie')
sigma = np.array([Sigma_zink.n, Sigma_gallium.n, Sigma_brom.n, Sigma_rubidium.n, Sigma_strontium.n, Sigma_zirkonium.n ])
Ord = np.array([30, 31, 35, 37, 38, 40]) 
E_k = np.array([Ezink.n, Egallium.n, Ebrom.n, Erubidium.n, Estrontium.n, Ezirkonium.n])
Ek_plot = np.sqrt(E_k)
xplot = np.linspace(90, 140, 1000)

params , ma = np.polyfit(Ek_plot ,Ord, deg =1, cov = True)
errors = np.sqrt(np.diag(ma))
a_ryd = ufloat(params[0], errors[0])
c_ryd = ufloat(params[1], errors[1])
print('1/sqrt(Rh)', a_ryd)
print('c', c_ryd)
Ryd = 1/a_ryd**2
print('R', Ryd)
print('dif', 1 - 13.6/Ryd.n)

##PLOTS
plt.figure()
plt.plot(Ek_plot, Ord, 'k.', label='Errechnete Punkte')
plt.plot(xplot, a_ryd.n * xplot + c_ryd.n, 'r-', label='Ausgleichsgerade')
plt.ylabel(r'$Z$')
plt.xlabel(r'$ \sqrt{E_{\text{K}}} \quad \mathbin{/} \si{\electronvolt}^{1/2}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/rydberg.pdf')


plt.figure()
plt.scatter([18.27], [N_zinkmin], s=20, marker='x', color='orange', label=r'$I_{\text{min}}$')
plt.scatter([19.0], [N_zinkmax], s=20, marker='x', color='green', label=r'$I_{\text{max}}$')
plt.scatter([tzink.n], [I_zink], s=20, marker='x', color='red', label=r'$I_{\text{K}}$')
plt.plot(zinkplot, m_zink * zinkplot + n_zink, '-b', label='lineare Interpolation')
plt.plot(t_zink, N_zink, 'k.', label='Messwerte')
plt.xlabel(r'$ \theta \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/zink.pdf')

plt.figure()
plt.scatter([17.05], [N_galliummin], s=20, marker='x', color='orange', label=r'$I_{\text{min}}$')
plt.scatter([17.85], [N_galliummax], s=20, marker='x', color='green', label=r'$I_{\text{max}}$')
plt.scatter([tgallium.n], [I_gallium], s=20, marker='x', color='red', label=r'$I_{\text{K}}$')
plt.plot(galliumplot, m_gallium * galliumplot + n_gallium, '-b', label='lineare Interpolation')
plt.plot(t_gallium, N_gallium, 'k.', label='Messwerte')
plt.xlabel(r'$ \theta \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/gallium.pdf')

plt.figure()
plt.scatter([13.0], [N_brommin], s=20, marker='x', color='orange', label=r'$I_{\text{min}}$')
plt.scatter([13.55], [N_brommax], s=20, marker='x', color='green', label=r'$I_{\text{max}}$')
plt.scatter([tbrom.n], [I_brom], s=20, marker='x', color='red', label=r'$I_{\text{K}}$')
plt.plot(bromplot, m_brom * bromplot + n_brom, '-b', label='lineare Interpolation')
plt.plot(t_brom, N_brom, 'k.', label='Messwerte')
plt.xlabel(r'$ \theta \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/brom.pdf')

plt.figure()
plt.scatter([11.35], [N_rubidiummin], s=20, marker='x', color='orange', label=r'$I_{\text{min}}$')
plt.scatter([12.1], [N_rubidiummax], s=20, marker='x', color='green', label=r'$I_{\text{max}}$')
plt.scatter([trubidium.n], [I_rubidium], s=20, marker='x', color='red', label=r'$I_{\text{K}}$')
plt.plot(rubidiumplot, m_rubidium * rubidiumplot + n_rubidium, '-b', label='lineare Interpolation')
plt.plot(t_rubidium, N_rubidium, 'k.', label='Messwerte')
plt.xlabel(r'$ \theta \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/rubidium.pdf')

plt.figure()
plt.scatter([10.7], [N_strontiummin], s=20, marker='x', color='orange' , label=r'$I_{\text{min}}$')
plt.scatter([11.6], [N_strontiummax], s=20, marker='x', color='green', label=r'$I_{\text{max}}$')
plt.scatter([tstrontium.n], [I_strontium], s=20, marker='x', color='red', label=r'$I_{\text{K}}$')
plt.plot(strontiumplot, m_strontium * strontiumplot + n_strontium, '-b', label='lineare Interpolation')
plt.plot(t_strontium, N_strontium, 'k.', label='Messwerte')
plt.xlabel(r'$ \theta \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/strontium.pdf')

plt.figure()
plt.scatter([9.5], [N_zirkoniummin], s=20, marker='x', color='orange'  , label=r'$I_{\text{min}}$')
plt.scatter([10.4], [N_zirkoniummax], s=20, marker='x', color='green', label=r'$I_{\text{max}}$')
plt.scatter([tzirkonium.n], [I_zirkonium], s=20, marker='x', color='red', label=r'$I_{\text{K}}$')
plt.plot(zirkoniumplot, m_zirkonium * zirkoniumplot + n_zirkonium, '-b', label='lineare Interpolation')
plt.plot(t_zirkonium, N_zirkonium, 'k.', label='Messwerte')
plt.xlabel(r'$ \theta \quad \mathbin{/} \si{\degree}$')
plt.ylabel(r'$ N \quad \mathbin{/} \si{Imp\per\second}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/zirkonium.pdf')