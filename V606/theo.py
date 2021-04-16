import numpy as np 



#chi_t = (m0 * (mb)**2 * (gj)**2 * N * J * (J+1) )/(3 * kb * T)

m0 = 1.257*10**(-6)
mb = ((-1.602*10**(-19))*6.626*10**(-34))/(4*np.pi * 9.109*10**(-31)) 
kb = 1.380 * 10**(-23)
T = 17 + 273.15 

#gj = (3*J*(J+1)+ (S*(S+1)-L*(L+1)))/(2*J*(J+1))

# Dy2O3
print('Dy2O3')
J = 7.5
S = 2.5
L = 5

M = 372.9982 #molare masse g/mol
dichte = 7.8 #g/cm^3
N = dichte/M #1/cm^3        SOLL DIE DICHTE IN M^3 ? UND KG ODER G?

gj = (3*J*(J+1)+ (S*(S+1)-L*(L+1)))/(2*J*(J+1))
print('g_j:', gj )

chi_t = (m0 * (mb)**2 * (gj)**2 * N * J * (J+1) )/(3 * kb * T)
print('chi_T:', chi_t)

# Gd2O3
print('Gd2O3')
J = 3.5
S = 3.5
L = 0

M = 362.4982 #molare masse g/mol
dichte = 7.4 #g/cm^3
N = dichte/M #1/cm^3 

gj = (3*J*(J+1)+ (S*(S+1)-L*(L+1)))/(2*J*(J+1))
print('g_j:', gj )

chi_t = (m0 * (mb)**2 * (gj)**2 * N * J * (J+1) )/(3 * kb * T)
print('chi_T:', chi_t)
