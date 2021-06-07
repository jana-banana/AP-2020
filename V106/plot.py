import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

import os

if os.path.exists("build") == False:
    os.mkdir("build")

print('g', const.g)

def Ttow(T):
    return 2*np.pi / T

def T_tplus(l):
    return 2 * np.pi * np.sqrt(l/const.g)

def w_tplus(l):
    return np.sqrt(const.g / l)

def K(a,b):
    z = a**2 - b**2
    n = a**2 + b**2
    return z/n

def w_tminus(l, K):
    return unp.sqrt((const.g/l) + (2*K/l) )

def T_tminus(l, K):
    return 2 * np.pi * unp.sqrt(l / (const.g + 2 * K))

def T_tschweb(a,b):
    z = a*b
    n = a-b
    return z/n

def diff(theo, exp):
    p = (1 - (exp/theo))
    return p*100

###Länge kurz: 35 cm
print('l = 0.35')
l = 0.35 #meter

t_rechts, t_links, t_gegen = np.genfromtxt('data/data_kurz.txt', unpack=True)
t_schwing, t_schweb = np.genfromtxt('data/data_kurz_gekoppelt.txt', unpack=True) 

#schwingungsdauern anpassen
t_rechts /= 5
t_links  /= 5
t_gegen  /= 5
t_schwing /= 2

#Mittelwerte errechnen
T_rechts = ufloat(np.mean(t_rechts), np.std(t_rechts)/np.sqrt(len(t_rechts)))

T_links  = ufloat(np.mean(t_links) , np.std(t_links) /np.sqrt(len(t_links )))

T_gleich = (T_rechts + T_links)/2

T_gegen  = ufloat(np.mean(t_gegen) , np.std(t_gegen) /np.sqrt(len(t_gegen )))

T_schwing = ufloat(np.mean(t_schwing), np.std(t_schwing)/np.sqrt(len(t_schwing)))

T_schweb  = ufloat(np.mean(t_schweb) , np.std(t_schweb) /np.sqrt(len(t_schweb )))

#Kreisfrequenz berechnen

w_gleich = Ttow(T_gleich)
w_gegen = Ttow(T_gegen)
w_schweb = Ttow(T_schweb)


#Kopplungskonstante berechnen
constK = K(T_gleich, T_gegen)

print('T_rechts', T_rechts)
print('T_links', T_links)
print('T_gleich', T_gleich)
print('T_gegen', T_gegen)
print('T_schwing', T_schwing)
print('T_schweb', T_schweb)
print('K', constK)

print('w_gleich', w_gleich)
print('w_gegen', w_gegen)
print('w_schweb', w_schweb)

#Theoriewerte berechnen
T_tp = T_tplus(l)
T_tm = T_tminus(l, constK)
T_tw = T_tschweb(T_tp, T_tm)

w_tp = w_tplus(l)
w_tm = w_tminus(l, constK)
w_tw = w_tp - w_tm


print('theo T_+:', T_tp)
print('theo w_+:', w_tp)
print(' ')
print('theo T_-:', T_tm)
print('theo w_-:', w_tm)
print(' ')
print('theo T_w:', T_tw)
print('theo w_w:', w_tw)
print(' ')


#Vergleich von exp und theo
print('Prozentuale Abweichung T_+', diff(T_tp, T_gleich))
print('Prozentuale Abweichung T_-', diff(T_tm, T_gegen))
print('Prozentuale Abweichung T_w', diff(T_tw, T_schweb))
print(' ')
print('Prozentuale Abweichung w_+', diff(w_tp, w_gleich))
print('Prozentuale Abweichung w_-', diff(w_tm,w_gegen))
print('Prozentuale Abweichung w_w', diff(w_tw,w_schweb))



###Länge lang: 102 cm
print('l = 1.02')
l = 1.02 #meter

t_rechts, t_links, t_gegen = np.genfromtxt('data/data_lang.txt', unpack=True)
t_schwing, t_schweb = np.genfromtxt('data/data_lang_gekoppelt.txt', unpack=True) 

#schwingungsdauern anpassen
t_rechts /= 5
t_links  /= 5
t_gegen  /= 5
t_schwing /= 2

#Mittelwerte errechnen
T_rechts = ufloat(np.mean(t_rechts), np.std(t_rechts)/np.sqrt(len(t_rechts)))

T_links  = ufloat(np.mean(t_links) , np.std(t_links) /np.sqrt(len(t_links )))

T_gleich = (T_rechts + T_links)/2

T_gegen  = ufloat(np.mean(t_gegen) , np.std(t_gegen) /np.sqrt(len(t_gegen )))

T_schwing = ufloat(np.mean(t_schwing), np.std(t_schwing)/np.sqrt(len(t_schwing)))

T_schweb  = ufloat(np.mean(t_schweb) , np.std(t_schweb) /np.sqrt(len(t_schweb )))

#Kreisfrequenz berechnen

w_gleich = Ttow(T_gleich)
w_gegen = Ttow(T_gegen)
w_schweb = Ttow(T_schweb)



#Kopplungskonstante berechnen
constK = K(T_gleich, T_gegen)

print('T_rechts', T_rechts)
print('T_links', T_links)
print('T_gleich', T_gleich)
print('T_gegen', T_gegen)
print('T_schwing', T_schwing)
print('T_schweb', T_schweb)
print('K', constK)

print('w_gleich', w_gleich)
print('w_gegen', w_gegen)
print('w_schweb', w_schweb)


#Theoriewerte berechnen
T_tp = T_tplus(l)
T_tm = T_tminus(l, constK)
T_tw = T_tschweb(T_tp, T_tm)

w_tp = w_tplus(l)
w_tm = w_tminus(l, constK)
w_tw = w_tp - w_tm


print('theo T_+:', T_tp)
print('theo w_+:', w_tp)
print(' ')
print('theo T_-:', T_tm)
print('theo w_-:', w_tm)
print(' ')
print('theo T_w:', T_tw)
print('theo w_w:', w_tw)
print(' ')


#Vergleich von exp und theo
print('Prozentuale Abweichung T_+', diff(T_tp, T_gleich))
print('Prozentuale Abweichung T_-', diff(T_tm, T_gegen))
print('Prozentuale Abweichung T_w', diff(T_tw, T_schweb))
print(' ')
print('Prozentuale Abweichung w_+', diff(w_tp, w_gleich))
print('Prozentuale Abweichung w_-', diff(w_tm,w_gegen))
print('Prozentuale Abweichung w_w', diff(w_tw,w_schweb))