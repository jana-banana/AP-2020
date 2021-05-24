import numpy as np 
import matplotlib.pyplot as plt 
import uncertainties.unumpy as unp 
from scipy.optimize import curve_fit

#eckiger stab
L0 = np.array([60, 60, 60.1, 60.1, 60])
L_e = np.mean(L0)
dL = np.std(L0)/5
print('L ist gleich', L_e ,'mit Fehler des Mittelwertes', dL )
L = unp.uarray(L_e, dL)
print(L)

b0 = np.array([1.03, 1.06, 1.01, 1.05, 1.03])
b_e = np.mean(b0)
db = np.std(b0)/5
print('b ist gleich', b_e ,'mit Fehler des Mittelwertes', db)
b = unp.uarray(b_e, db)*(10**(-2))
print(b)


d0 = np.array([1.01, 1.02, 1.01, 1.01, 1.03])
d_e = np.mean(d0)
dd = np.std(d0)/5
print('d ist gleich', d_e ,'mit Fehler des Mittelwertes', dd)
d = unp.uarray(d_e, dd)*(10**(-2))
print(d)

x, D_0, D, deltaD = np.genfromtxt('data1k.txt', unpack=True)
L_e = 600.4
x_plot = (L_e*(x**2) - (1/3)*(x**3))*10**(-6)
print('x_plot =', x_plot)
#curve-fit
def line(X, A, B):
    return A * X + B

popt, pcov = curve_fit(line, x_plot,deltaD )
errors = np.sqrt(np.diag(pcov))

a = unp.uarray(popt[0],errors[0])
c = unp.uarray(popt[1], errors[1])
print('a=', a)
print('c=', c)
I = (1/24)*(b*d)*(b**2 + d**2)
print('I =', I)

m=1.1003
E = (m * 9.81)/(2* I * a*(10**6))
print('E=', E)


#runder stab 
L0 = np.array([60, 60, 60, 60.1, 60.05])
L_e = np.mean(L0)
dL = np.std(L0)/5
print('L ist gleich', L_e ,'mit Fehler des Mittelwertes', dL )
L = unp.uarray(L_e, dL)
print(L)

d0 = np.array([1.1, 1.09, 1.08, 1.09, 1.09])
d_e = np.mean(d0)
dd = np.std(d0)/5
print('d ist gleich', d_e ,'mit Fehler des Mittelwertes', dd)
print('r ist gleich', d_e/2 ,'mit Fehler des Mittelwertes ', np.std(d0/2)/5)
r = unp.uarray(d_e/2, np.std(d0/2)/5)
print(r)

x, D_0, D, deltaD = np.genfromtxt('data1m.txt', unpack=True)
L_e = 600.3
x_plot = (L_e*(x**2) - (1/3)*(x**3))
print('x_plot =', x_plot)
#curve-fit
def line(X, A, B):
    return A * X + B

popt, pcov = curve_fit(line, x_plot,deltaD )
errors = np.sqrt(np.diag(pcov))

a = unp.uarray(popt[0],errors[0])
c = unp.uarray(popt[1], errors[1])
print('a=', a)
print('c=', c)

I = (np.pi * (r*10**(-2))**4)/4
print('I =', I)

m=0.5996
E = (m * 9.81)/(2* I * (a*10**6))
print('E=', E)

#beidseitige einspannung
print('L = ', L)
print('r =', r)

x1, D_01, D1, deltaD1= np.genfromtxt('data2m.txt', unpack=True)
L_e = 600.3
x_plot1 = (3*(L_e**2) *x1 - 4*(x1**3))*10**(-6)
print('x_plot1 =', x_plot1)

#curve-fit
def line(X, A, B):
    return A * X + B

popt1, pcov1 = curve_fit(line, x_plot1,deltaD1 )
errors = np.sqrt(np.diag(pcov1))
print(errors)

a1 = unp.uarray(popt1[0], errors[0])
b1 = unp.uarray(popt1[1], errors[1])
print('a1=', a1)
print('b1=', b1)

I = (np.pi * (r*10**(-2))**4)/4
print('I =', I)

m=1.0995
E = (m * 9.81)/(48* I * a1)
print('E=', E)

x2, D_02, D2, deltaD2 = np.genfromtxt('data2m2.txt', unpack=True)

x_plot2 = (4* x2**3 - 12*L_e * x2**2 + 9*(L_e**2)*x2 - L_e**3)*10**(-6)
print('x_plot2 = ', x_plot2)
#curve-fit
def line(X, A, B):
    return A * X + B

popt2, pcov2 = curve_fit(line, x_plot2,deltaD2 )
errors2 = np.sqrt(np.diag(pcov2))
print(errors2)

a2 = unp.uarray(popt2[0] , errors2[0])
c2 = unp.uarray(popt2[1] , errors2[1] )
print('a2= ', a2)
print('c2= ', c2)

I = (np.pi * (r*10**(-2))**4)/4
print('I =', I)

m=1.0995
E = (m * 9.81)/(48* I * a2)
print('E=', E)