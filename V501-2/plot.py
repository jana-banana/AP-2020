import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

# Elektronen im E-FELD
# Aufgabe A
################ Empfindlichkeiten D/U_d bzw. a ermitteln
################################### U = 180V #######################
D_180, U_d_180 = np.genfromtxt('data/data_180V.txt', unpack=True)
D_180 *= 6.35 # mm
D_180 -= 6.35*1.75 # minus nullauslenkung

plt.plot(D_180, U_d_180,'b.', label='$U_B = 180$V')

plt.xlabel('$U_d  $ / V')
plt.ylabel('$D  $ / cm')

def line(x, a, b):
    return a * x + b

popt, pcov = curve_fit(line, D_180, U_d_180)

print("a =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)

xfine = np.linspace(-40., 20.)  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'b-')

################################### U = 240 V #######################
D_240, U_d_240 = np.genfromtxt('data/data_240V.txt', unpack=True)
D_240 *= 6.35 # mm
D_240 -= 6.35*1.75 # minus nullauslenkung

plt.plot(D_240, U_d_240,'c.', label='$U_B = 240$V')

popt, pcov = curve_fit(line, D_240, U_d_240)

print("a =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)

xfine = np.linspace(-40., 20.)  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'c-')
################################### U = 275 V #######################
D_275, U_d_275 = np.genfromtxt('data/data_275V.txt', unpack=True)
D_275 *= 6.35 # mm
D_275 -= 6.35*1.75 # minus nullauslenkung

plt.plot(D_275, U_d_275,'m.', label='$U_B = 275$V')

popt, pcov = curve_fit(line, D_275, U_d_275)

print("a =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)

xfine = np.linspace(-40., 20.)  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'm-')
################################### U = 300 V #######################
D_300, U_d_300 = np.genfromtxt('data/data_300V.txt', unpack=True)
D_300 *= 6.35 # mm
D_300 -= 6.35*1.75 # minus nullauslenkung

plt.plot(D_300, U_d_300,'y.', label='$U_B = 300$V')

popt, pcov = curve_fit(line, D_300, U_d_300)

print("a =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)

xfine = np.linspace(-40., 20.)  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'y-')
################################### U = 350 V #######################
D_350, U_d_350 = np.genfromtxt('data/data_350V.txt', unpack=True) 
D_350 *= 6.35 # mm
D_350 -= 6.35*1.75 # minus nullauslenkung

plt.plot(D_350, U_d_350,'g.', label='$U_B = 350$V')

popt, pcov = curve_fit(line, D_350, U_d_350)

print("a =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)

xfine = np.linspace(-40., 20.)  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'g-')
###################################################################

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('bilder/E_Feld_Teil_1.pdf')
plt.clf()

############## Empfindlichkeiten gegen 1/U_b auftragen und dann nochmal steigung bestimmen
############################bis hier läuft#################
# a = D/U_d die Empfindlichkeit von vorher

# a gegen 1/U_B auftragen, a ist y Achse, hoffentlich, jo ist ok

#a = np.array([ufloat(-0.5371391075406825, 0.005966852273172693),  # 180V
#                 ufloat(-0.7039370079653217, 0.004787864225297844),  # 240V
#                 ufloat(-0.7805774226442466, 0.004119029471558086),  # 275V
#                 ufloat(-0.6900262439813777, 0.08036888027272444),   # 300V
#                 ufloat(-0.9770626499279077, 0.017260524997896702)] ) # 350V

A = [-0.5371391075406825, -0.7039370079653217, -0.7805774226442466, -0.6900262439813777, -0.9770626499279077]
A_ausgleich = [-0.5371391075406825, -0.7039370079653217, -0.7805774226442466, -0.9770626499279077]

errA = [0.005966852273172693, 0.004787864225297844, 0.004119029471558086, 0.08036888027272444, 0.017260524997896702]
#errA = [0.005966852273172693, 0.004787864225297844, 0.004119029471558086, 0.017260524997896702]

U_B = [1/180, 1/240, 1/275, 1/300, 1/350]
U_B_ausgleich = [1/180, 1/240, 1/275, 1/350]


plt.plot( U_B, A, '.',label= 'U_b gegen a')
plt.errorbar( U_B, A, yerr = errA, fmt='o')

plt.ylabel('$D/U_d$ /  mm/V')
plt.xlabel('$1/U_B$ /  1/V')

popt, pcov = curve_fit(line, U_B_ausgleich, A_ausgleich)

print("D/U_d gegen 1/U_b")
print("a =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)

xfine = np.linspace(0.001, 0.006)  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), '-')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('bilder/E_Feld_Teil_1_2.pdf')
plt.clf()

p = 19  # mm Ablenkplatte
d = 3.8 # mm abstand der Ablenkplatte
L = 143 # mm Abstand ablenkplatte und Leuchtschirm

insgesamt = (p*L)/(2*d)
print('(p*L)/(2*d) :', insgesamt)
print('abweichung: a/(p*L)/(2*d)',popt[0] / insgesamt)

################################################################

# Elektronen im B Feld
print('Elektronen im B Feld U_b = 250')

D_b_250, I_250 = np.genfromtxt('data/data_b250V.txt', unpack=True)
D_b_250 *= 0.00635 # m
#D_250 -= 6.35*1.75 # minus nullauslenkung

N = 20   # windungszahl
R = 0.282 #spulenradius m
L = 0.143  # Abstand ablenkplatte und Leuchtschirm m
mu_0 = 4* np.pi * 10**(-7) # Vs/Am
D_bruch = D_b_250/(L**2 + D_b_250**2)
B = (mu_0 * 8 * N * I_250)/(np.sqrt(125) * R) *10**3#Tesla

plt.plot( B, D_bruch,'r.', label='$U_B = 250 $V')

print(B, D_bruch)

popt, pcov = curve_fit(line, B, D_bruch)

print('B-Feld teil 2')
print("a*10^-3 =", popt[0], "+/-", pcov[0,0]**0.5)
print("b ohne 10^3, also so ok =", popt[1], "+/-", pcov[1,1]**0.5)

#xfine = np.linspace(0., 1.7*10**(-7) ) # define values to plot the function for
#plt.plot(xfine, line(xfine, popt[0], popt[1]), '-')
plt.plot(B, line(B, popt[0], popt[1]), 'r-')


# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig('bilder/B_Feld_Teil_1_180.pdf')
# plt.clf()

a = ufloat(popt[0]*10**3, (pcov[0,0]*10**3)**0.5)
e_m_250 = (a * np.sqrt(8 * 250))**2
print('e_m_250: ',e_m_250)

print('Elektronen im B Feld U_b = 420')

D_b_420, I_420 = np.genfromtxt('data/data_b420V.txt', unpack=True)
D_b_420 *= 0.00635 # mm
#D_250 -= 6.35*1.75 # minus nullauslenkung

N = 20   # windungszahl
R = 0.282 #spulenradius m
L = 0.143  # Abstand ablenkplatte und Leuchtschirm m
mu_0 = 4* np.pi * 10**(-7) # Vs/Am 
D_bruch = D_b_420/(L**2 + D_b_420**2)
B = (mu_0 * 8 * N * I_420)/(np.sqrt(125) * R)*10**3  #Tesla

plt.plot( B, D_bruch,'b.', label='$U_B = 420 $V')

print(B, D_bruch)

plt.xlabel('$ B $ / mT')
plt.ylabel('$D/(L^2 + D^2) $ / 1/m')

popt, pcov = curve_fit(line, B, D_bruch)


print("a* 10^-3 =", popt[0], "+/-", pcov[0,0]**0.5)
print("b so ok =", popt[1], "+/-", pcov[1,1]**0.5)

#xfine = np.linspace(0., 1.7*10**(-7) ) # define values to plot the function for
#plt.plot(xfine, line(xfine, popt[0], popt[1]), '-')
plt.plot(B, line(B, popt[0], popt[1]), 'b-')


plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('bilder/B_Feld_Teil_1_beide.pdf')
plt.clf()

a = ufloat(popt[0]*10**3, (pcov[0,0]*10**3)**0.5)
e_m_420 = (a * np.sqrt(8 * 250))**2
print('e_m_420: ',e_m_420) #besser mit uncert.

####################################### Teil 2
print('erdmagnetfeld')
B_erde = (mu_0 * 8 * N * 0.55)/(np.sqrt(125) * R) #Tesla?
print(B_erde)

print('mittelwert von e0_m0, ' , (e_m_250 +e_m_420)/2)