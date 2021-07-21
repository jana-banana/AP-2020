import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

import os

if os.path.exists("build") == False:
    os.mkdir("build")

if os.path.exists("build/data") == False:
    os.mkdir("build/data")
#--------------------------------------------------------------------------------------------------------------------------
###allgemeine Definitionen
def diff(exp, theo):
    return (1 - (exp / theo))*100 

#brechungsindices
n_luft = 1.000292
n_wasser = 1.33
n_kronglas = 1.46 #bis 1.65
n_plexiglas = 1.49
n_diamant = 2.42 

#laser lambda
grun = 532e-9 #metre
rot = 635e-9 #metre

#------------------------------------------------------------------------------------------------------------------------------------------------------------
###Relexionsgesetz
print('Reflexionsgesetz')

a = np.array([20 , 30, 35, 40, 50, 60, 70])
b = np.array([20 , 30.5, 36, 41, 51, 61.5, 72])

#lineare Ausgleichsrechnung
params, cov = np.polyfit(a , b , deg=1, cov=True)
errs = np.sqrt(np.diag(cov))
for name, value, error in zip('AB', params, errs):
    print(f'{name} = {value:.3f} +- {error:.3f}')

print('dif A:', diff(params[0], 1))

#plot
x_plot = np.linspace(20, 71, 1000)

plt.figure()
plt.plot(a, b, '.k', label='Messwerte')
plt.plot(x_plot, params[0]*x_plot + params[1], '-g', label='Ausgleichsgerade')
plt.xlabel(r'$\alpha \, \mathbin{/} \si{\degree} $')
plt.ylabel(r'$\beta \, \mathbin{/} \si{\degree} $')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/reflexion.pdf')

#für Tabellen
np.savetxt(
    'build/data/reflexion.txt',
    np.column_stack([a,b]),
    fmt=['%.1f', '%.1f'],       
    delimiter=' & ',
    header='alpha, beta',
)

print(' ')
#------------------------------------------------------------------------------------------------------------------------------------------------------------
###Brechungsgesetz
print('Brechungsgesetz')

alpha = np.array([20, 30, 35, 40, 50, 60, 70])
beta  = np.array([14, 20, 23, 26, 31.5, 36, 39])

#brechungsindex für plexiglas bestimmen
def brechn(a, b):
    return np.sin(a * np.pi / 180 ) / np.sin(b*np.pi/180)

n_ar = brechn(alpha, beta)
print('Array für n', n_ar)

n = unp.uarray( np.mean(n_ar), np.std(n_ar)/np.sqrt(len(n_ar)))
print('Mittelwert für n', n)

print('diff in %', diff(unp.nominal_values(n), n_plexiglas))

#Lichtgeschwidigkeit in Plexiglas
v_plexi = const.c / n 
v_theo = const.c / n_plexiglas
d_plexi = diff(v_plexi, v_theo)
print(f'v_plexiglas: {v_plexi:.3f} \n v_theo: {v_theo:.3f} \n Differenz in % {d_plexi:.3f}')

#für Tabellen
np.savetxt(
    'build/data/brechungsgesetz.txt',
    np.column_stack([alpha,beta, n_ar]),
    fmt=['%.1f', '%.1f', '%.3f'],       
    delimiter=' & ',
    header='alpha, beta, n',
)

print(' ')
#------------------------------------------------------------------------------------------------------------------------------------------------------------
###Planparallele Platten
print('Planparallele Platten')

d = 0.0585 #metre

#strahlenversatz
def s(a,b):
    return d* unp.sin((a-b)*np.pi/180) / unp.cos(b*np.pi/180)

#version 1
s1_ar = s(alpha ,beta )
s1 = unp.uarray(np.mean(s1_ar), np.std(s1_ar)/np.sqrt(len(s1_ar)) )

#version 2
beta2 = unp.arcsin(np.sin(alpha * np.pi /180)/ unp.nominal_values(n)) * 180 / np.pi #degree
s2_ar = s(alpha, beta2)
s2 = unp.uarray(np.mean(s2_ar), np.std(s2_ar)/np.sqrt(len(s2_ar)) )

print(f'Version 1 \n s = {s1*10**3}mm \n Version 2 \n s = {s2*10**3}mm ')

print('Unterschied mit s1= theo', diff(s2, s1))
print('Unterschied mit s2= theo', diff(s1, s2))

#für Tabellen
np.savetxt(
    'build/data/planparallel.txt',
    np.column_stack([alpha,beta, s1_ar*10**3, beta2, s2_ar*10**3]),
    fmt=['%.1f', '%.1f', '%.3f', '%.3f', '%.3f'],       
    delimiter=' & ',
    header='alpha, beta, s1, beta2, s2',
)

print(' ')
#------------------------------------------------------------------------------------------------------------------------------------------------------------
###Prisma
print('Prisma')

gamma = 60 #degree

def delta(a1, a2, b1, b2):
    return (a1 + a2) - (b1 + b2)

#Daten einlesen und bearbeiten
a1 = np.array([30, 35, 40, 50, 60])
a2_g = np.array([78.5, 66.5, 59, 47.5, 38.5])
a2_r = np.array([78, 65.5, 58, 47, 37.5])

b1 = unp.arcsin(np.sin(a1 * np.pi /180)/ n_kronglas) * 180 / np.pi #degree
b2_g = unp.arcsin(np.sin(a2_g * np.pi /180)/ n_kronglas) * 180 / np.pi #degree
b2_r = unp.arcsin(np.sin(a2_r * np.pi /180)/ n_kronglas) * 180 / np.pi #degree
#prüfen
print('prüfen:', b1 + b2_g - 60, b1 + b2_r - 60)

#Deltas berechnen
delta_g = delta(a1, a2_g, b1, b2_g)
delta_r = delta(a1, a2_r, b1, b2_r)

#plot, wieso denn auch nicht ?
plt.figure()
plt.plot(a2_g, delta_g, '.g', label='Ablenkung für grünes Licht')
plt.plot(a2_r, delta_r, '.r', label='Ablenkung für rotes Licht')
plt.xlabel(r'Ablenkung $\delta \, \mathbin{/} \si{\degree} $')
plt.xlabel(r'Einfallswinkel $\alpha_1 \, \mathbin{/} \si{\degree} $')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/prisma.pdf')

#für Tabellen
np.savetxt(
    'build/data/prisma.txt',
    np.column_stack([a1 ,b1, a2_g, b2_g,delta_g, a2_r, b2_r, delta_r]),
    fmt=['%.1f', '%.3f', '%.1f', '%.3f','%.3f' , '%.1f','%.3f', '%.3f'],       
    delimiter=' & ',
    header='a1 ,b1, a2_g, b2_g,delta_g, a2_r, b2_r, delta_r',
)

print(' ')
#------------------------------------------------------------------------------------------------------------------------------------------------------------
###Beugung am Gitter
print('Beugung am Gitter')

d1 = 5/3e6 #metre
d2 = 10/3e6 #metre
d3 = 10e-6 #metre

#daten einlesen - wahr das sinnvoll ?

# #d1
# p_g = np.array([-19.5, 18]) #k=1
# p_r = np.array([-23, 22]) #k=1
# #d2
# ph_g = np.array([-9.5, 8.5, -19, 17.5, -29,  27.5])
# ph_r = np.array([-11, 10.5, -23, 21.5, -35,  34])
# #d3 
# phi_g = np.array([-3.5, 2.5, -7, 5.5, -10, 8.5, -13, 11.5, -16.5, 15, -19.5, 18, -23, 21.5, -26.5, 24.5, -30, 28])
# phi_r = np.array([-4, 3.5, -8, 7, -11.5, 10.5, -15.5, 14.5, -19.5, 18, -23.5, 22, -27.5, 26, 32, 30])

#d1
p_g = np.array([19.5, 18]) #k=1
p_r = np.array([23, 22]) #k=1
#d2
ph_g = np.array([9.5, 8.5, 19, 17.5, 29,  27.5])
ph_r = np.array([11, 10.5, 23, 21.5, 35,  34])
#d3 
phi_g = np.array([3.5, 2.5, 7, 5.5, 10, 8.5, 13, 11.5, 16.5, 15, 19.5, 18, 23, 21.5, 26.5, 24.5, 30, 28])
phi_r = np.array([4, 3.5, 8, 7, 11.5, 10.5, 15.5, 14.5, 19.5, 18, 23.5, 22, 27.5, 26, 32, 30])

def lamb(phi, d, k):
    return d*(np.sin(phi*np.pi/180)/k)


#lass uns lambdas rechnen
lambg_ar = np.array([lamb(p_g[0], d1, 1), lamb(p_g[1], d1, 1), lamb(ph_g[0], d2, 1),  lamb(ph_g[1], d2, 1),  lamb(ph_g[2], d2, 2),  lamb(ph_g[3], d2, 2),  lamb(ph_g[4], d2, 3) ,  lamb(ph_g[5], d2, 3), 
                     lamb(phi_g[0], d3, 1), lamb(phi_g[1], d3, 1), lamb(phi_g[2], d3, 2), lamb(phi_g[3], d3, 2), lamb(phi_g[4], d3, 3), lamb(phi_g[5], d3, 3), lamb(phi_g[6], d3, 4), lamb(phi_g[7], d3, 4),
                     lamb(phi_g[8], d3, 5), lamb(phi_g[9], d3, 5), lamb(phi_g[10], d3, 6), lamb(phi_g[11], d3, 6), lamb(phi_g[12], d3, 7), lamb(phi_g[13], d3, 7), lamb(phi_g[14], d3, 8), lamb(phi_g[15], d3, 8), 
                     lamb(phi_g[16], d3, 9), lamb(phi_g[17], d3, 9)])

lambr_ar = np.array([lamb(p_r[0], d1, 1), lamb(p_r[1], d1, 1), lamb(ph_r[0], d2, 1),  lamb(ph_r[1], d2, 1),  lamb(ph_r[2], d2, 2),  lamb(ph_r[3], d2, 2),  lamb(ph_r[4], d2, 3) ,  lamb(ph_r[5], d2, 3), 
                     lamb(phi_r[0], d3, 1), lamb(phi_r[1], d3, 1), lamb(phi_r[2], d3, 2), lamb(phi_r[3], d3, 2), lamb(phi_r[4], d3, 3), lamb(phi_r[5], d3, 3), lamb(phi_r[6], d3, 4), lamb(phi_r[7], d3, 4),
                     lamb(phi_r[8], d3, 5), lamb(phi_r[9], d3, 5), lamb(phi_r[10], d3, 6), lamb(phi_r[11], d3, 6), lamb(phi_r[12], d3, 7), lamb(phi_r[13], d3, 7), lamb(phi_r[14], d3, 8), lamb(phi_r[15], d3, 8)])

print('lambda grün', lambg_ar)
print('lambda rot', lambr_ar)

#mittelwert
lambd_g = unp.uarray(np.mean(lambg_ar), np.std(lambg_ar)/np.sqrt(len(lambg_ar)))
lambd_r = unp.uarray(np.mean(lambr_ar), np.std(lambr_ar)/np.sqrt(len(lambr_ar))) 

d_g = diff(lambd_g, grun)
d_r = diff(lambd_r, rot)

print(f'Grün {lambd_g*10**9}nm  theo {grun}\n Abweichung in % {d_g:.3f} \n Rot {lambd_r*10**9}nm theo {rot}\n Abweichung in % {d_r:.3f}')