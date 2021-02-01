import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit


Nun = np.array([129, 143, 144, 136, 139, 126, 158])
Nu30n = np.mean((Nun/10))
Nu30s = np.std((Nun/10))/np.sqrt(7)
Nu30 = unp.uarray(Nu30n , Nu30s)
print('Nu30',Nu30)

dt, Ngesn = np.genfromtxt('vanadium.txt', unpack=True)
Nges = unp.uarray(Ngesn, np.sqrt(Ngesn))
N = Nges - Nu30

t_plot = np.linspace(25 , 1400, 1000)

#curve-fit
def line(X, A, B):
    return A * X + B

popt, pcov = curve_fit(line, dt, np.log10(unp.nominal_values(N)))
errors = np.sqrt(np.diag(pcov))

a = unp.uarray(popt[0],errors[0])
c = unp.uarray(popt[1], errors[1])
print('a=', a)
print('c=', c)

umrechnung = np.log10(np.exp(1))
Steigung = a / umrechnung #steigung in ln
ce = c / umrechnung
halbwertszeit = np.log(2)/(-Steigung)
print('ae', Steigung)
print('ce', ce)
print('Halbwertszeit', halbwertszeit)

plt.errorbar(dt, unp.nominal_values(N), yerr=unp.std_devs(N), fmt='k.', label='Vanadium')
plt.plot(t_plot, 10**(popt[0]*t_plot + popt[1]), label='Ausgleichsgerade')
plt.xlabel(r'$t [\text{s}]$')
plt.ylabel('Anzahl der Impulse')
plt.yscale('log')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/vanadium.pdf')

plt.clf()

#nur ein teil vanadium
dt, Ngesn = np.genfromtxt('vana2.txt', unpack=True)
Nges = unp.uarray(Ngesn, np.sqrt(Ngesn))
N = Nges - Nu30

t_plot = np.linspace(25 , 450, 1000)

#curve-fit
def line(X, A, B):
    return A * X + B

popt, pcov = curve_fit(line, dt, np.log10(unp.nominal_values(N)))
errors = np.sqrt(np.diag(pcov))

a = unp.uarray(popt[0],errors[0])
c = unp.uarray(popt[1], errors[1])
print('a=', a)
print('c=', c)

umrechnung = np.log10(np.exp(1))
Steigung = a / umrechnung #steigung in ln
halbwertszeit = np.log(2)/(-Steigung)
print('ae', Steigung)
print('Halbwertszeit', halbwertszeit)

plt.errorbar(dt, unp.nominal_values(N), yerr=unp.std_devs(N), fmt='k.', label='Messwerte')
plt.plot(t_plot, 10**(popt[0]*t_plot + popt[1]), label='Ausgleichsgerade')
plt.xlabel(r'$t [\text{s}]$')
plt.ylabel('Anzahl der Impulse')
plt.yscale('log')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/vana2.pdf')

plt.clf()

#####Rhodium
Nun = np.array([129, 143, 144, 136, 139, 126, 158])
Nu15n = np.mean((Nun/20))
Nu15s = np.std((Nun/20))/np.sqrt(7)
Nu15 = unp.uarray(Nu15n , Nu15s)
print('Nu15',Nu15)

dt, Ngesn = np.genfromtxt('rodium.txt', unpack=True)
Nges = unp.uarray(Ngesn, np.sqrt(Ngesn))
N = Nges - Nu15
#print('Werte Rhodium',N)
t_plot = np.linspace(10, 700, 1000)

dtl, Ngesnl = np.genfromtxt('rhodium_lang.txt', unpack=True)
Ngesl = unp.uarray(Ngesnl, np.sqrt(Ngesnl))
Nl = Ngesl - Nu15

#curve-fit
def line(X, A, B):
    return A * X + B

poptl, pcov = curve_fit(line, dtl, np.log10(unp.nominal_values(Nl)))
errors = np.sqrt(np.diag(pcov))

a = unp.uarray(poptl[0],errors[0])
c = unp.uarray(poptl[1], errors[1])
print('a_lang=', a)
print('c_lang=', c)

umrechnung = np.log10(np.exp(1))
Steigung = a / umrechnung #steigung in ln
halbwertszeit = np.log(2)/(-Steigung)
print('ae_lang', Steigung)
print('Halbwertszeit_lang', halbwertszeit)

N_lang = unp.exp(c)*(1-unp.exp(-Steigung*15))
print('N_lang', N_lang)

plt.errorbar(dt, unp.nominal_values(N), yerr=unp.std_devs(N), fmt='k.', label='Messwerte')
plt.plot(t_plot, 10**(poptl[0]*t_plot + poptl[1]), label='Ausgleichsgerade')
plt.axvline(x=240, label="t*", color='m')
plt.xlabel(r'$t [\text{s}]$')
plt.ylabel('Anzahl der Impulse')
plt.yscale('log')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/rhodium.pdf')

plt.clf()
##rhodium kurz
print('Nu15', Nu15)
dtk, Ngesnk = np.genfromtxt('rhodium_kurz.txt', unpack=True)
Ngesk = unp.uarray(Ngesnk, np.sqrt(Ngesnk))
Nk = Ngesk - Nu15
def Nlf(t):
    return -0.36*np.exp(-0.0033*t)
tdl = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210])
Ndl = Nlf(tdl)
#print('N_lang(t)', Ndl)
Nkk = Nk - Ndl
#print('N_kurz für den Plot', Nkk)
t_plot = np.linspace(10, 220, 1000)

#curve-fit
def line(X, A, B):
    return A * X + B

poptk, pcov = curve_fit(line, dtk, np.log10(unp.nominal_values(Nkk)))
errors = np.sqrt(np.diag(pcov))

a = unp.uarray(poptk[0],errors[0])
c = unp.uarray(poptk[1], errors[1])
print('a_kurz=', a)
print('c_kurz=', c)

umrechnung = np.log10(np.exp(1))
Steigung = a / umrechnung #steigung in ln
halbwertszeit = np.log(2)/(-Steigung)
print('ae_kurz', Steigung)
print('Halbwertszeit_kurz', halbwertszeit)

plt.errorbar(dtk, unp.nominal_values(Nkk), yerr=unp.std_devs(Nkk), fmt='k.', label='Messwerte')
plt.plot(t_plot, 10**(poptk[0]*t_plot + poptk[1]), label='Ausgleichsgerade')
plt.xlabel(r'$t [\text{s}]$')
plt.ylabel('Anzahl der Impulse')
plt.yscale('log')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/rhodium_kurz.pdf')

plt.clf()
t_plot = np.linspace(10, 680)

plt.errorbar(dt, unp.nominal_values(N), yerr=unp.std_devs(N), fmt='k.', label='Messwerte')
plt.plot(t_plot, 10**(poptk[0]*t_plot + poptk[1]), label='Ausgleichsgerade für kurzlebigen Zerfall')
plt.plot(t_plot, 10**(poptl[0]*t_plot + poptl[1]), label='Ausgleichsgerade für langlebigen Zerfall')
plt.plot(t_plot, 10**(t_plot*(poptk[0]+poptl[0])+poptk[1]+poptl[1]), label='Summe aus beiden Ausgleichsgeraden')
plt.xlabel(r'$t [\text{s}]$')
plt.ylabel('Anzahl der Impulse')
plt.yscale('log')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/rhodium_ges.pdf')