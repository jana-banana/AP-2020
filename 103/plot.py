import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

x, D_0, D, deltaD = np.genfromtxt('data1m.txt', unpack=True)
L_e = 600.3

x_plot = (L_e*(x**2) - (1/3)*(x**3))*10**(-6)
plt.plot(x_plot, deltaD, 'k.', label='Messwerte')

#curve-fit
def line(X, A, B):
    return A * X + B

popt, pcov = curve_fit(line, x_plot,deltaD )
errors = np.sqrt(np.diag(pcov))
print(errors)

print("A =", popt[0], '+-', errors[0])
print("B =", popt[1], '+-' , errors[1])

x_curve = np.linspace(0, 110)
a = popt[0]
b = popt[1]
plt.plot(x_curve, a*x_curve + b, 'b-', label='Ausgleichsgerade')

plt.grid()
plt.legend()

plt.xlabel(r'$ L\cdot x^2- \frac{x^3}{3} \mathbin{/} \SI{e-3}{\metre\tothe{3}}$')
plt.ylabel(r'$ D \mathbin{/} \si{\milli\metre}$')
plt.tight_layout()
plt.savefig('build/plotm1.pdf')


plt.clf()
x, D_0, D, deltaD= np.genfromtxt('data1k.txt', unpack=True)
L_e = 600.4

x_plot = (L_e*(x**2) - (1/3)*(x**3))*10**(-6)
plt.plot(x_plot, deltaD, 'k.', label='Messwerte')

#curve-fit
def line(X, A, B):
    return A * X + B

popt, pcov = curve_fit(line, x_plot,deltaD )
errors = np.sqrt(np.diag(pcov))
print(errors)

print("A =", popt[0], '+-', errors[0])
print("B =", popt[1], '+-' , errors[1])

x_curve = np.linspace(0, 110)
a = popt[0]
b = popt[1]
plt.plot(x_curve, a*x_curve + b, 'm-', label='Ausgleichsgerade')

plt.grid()
plt.legend()
plt.gcf().subplots_adjust(bottom=0.25)

plt.xlabel(r'$L \cdot x^2 - \frac{x^3}{3} \mathbin{/} \SI{e-3}{\metre\tothe{3}}$')
plt.ylabel(r'$D \mathbin{/} \si{\milli\metre}$')
plt.tight_layout()
plt.savefig('build/plotk.pdf')


plt.clf()

x1, D_01, D1, deltaD1 = np.genfromtxt('data2m.txt', unpack=True)
L_e = 600.3

x_plot1 = (3*(L_e**2)*x1 - 4*(x1**3))*10**(-6)
plt.plot(x_plot1, deltaD1, 'k.', label='Messwerte')

#curve-fit
def line(X, A, B):
    return A * X + B

popt1, pcov1 = curve_fit(line, x_plot1,deltaD1 )
errors = np.sqrt(np.diag(pcov1))
print(errors)

x_curve1 = np.linspace(0, 220)
a1 = popt1[0]
b1 = popt1[1]

plt.plot(x_curve1, a1*x_curve1 + b1, 'r-', label='Ausgleichsgerade')

plt.grid()
plt.legend()

plt.xlabel(r'$(3L^2x - 4x^3) \mathbin{/} \SI{e-3}{\metre\tothe{3}}$')
plt.ylabel(r'$D \mathbin{/} \si{\milli\metre}$')
plt.tight_layout()
plt.savefig('build/plot2.pdf')

# zweiter plot
plt.clf()
x2, D_02, D2, deltaD2 = np.genfromtxt('data2m2.txt', unpack=True)
L_e = 600.3

x_plot2 = (4*(x2**3) - 12*L_e * x2**2 + 9*(L_e**2)*x2 - L_e**3)*10**(-6)
plt.plot(x_plot2, deltaD2, 'k.', label='Messwerte')

#curve-fit
def line(X, A, B):
    return A * X + B

popt2, pcov2 = curve_fit(line, x_plot2,deltaD2 )
errors2 = np.sqrt(np.diag(pcov2))
print(errors2)

x_curve2 = np.linspace(0, 220)
a2 = popt2[0]
b2 = popt2[1]

plt.plot(x_curve2, a2*x_curve2 + b2, 'r-', label='Ausgleichsgerade')

plt.grid()
plt.legend()

plt.xlabel(r'$(4x^3 - 12L x^2 + 9L^2x - L^3) \mathbin{/} \SI{e-3}{\metre\tothe{3}}$')
plt.ylabel(r'$D \mathbin{/} \si{\milli\metre}$')
plt.tight_layout()
plt.savefig('build/plot22.pdf')