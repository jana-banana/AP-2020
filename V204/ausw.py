import numpy as np 

t, T1, T4, T5, T8, T2, T3, T6, T7 = np.genfromtxt('data_stat.txt', unpack=True)

print('T1 von 700s:', T1[140],T1[140]+273.15)
print('T4 von 700s:', T4[140],T4[140]+273.15)
print('T5 von 700s:', T5[140],T5[140]+273.15)
print('T8 von 700s:', T8[140],T8[140]+273.15)

#nach Gleichung 1
#Messing breit

A = 0.012 *0.004 
k = 120 #watt pro meter kelvin
dQ_100 = -A*k*((T1[20]-T2[20])/0.03)
dQ_200 = -A*k*((T1[40]-T2[40])/0.03)
dQ_350 = -A*k*((T1[70]-T2[70])/0.03)
dQ_450 = -A*k*((T1[90]-T2[90])/0.03)
dQ_600 = -A*k*((T1[120]-T2[120])/0.03)
print('dQ:', dQ_100, dQ_200, dQ_350, dQ_450, dQ_600)

#messing schmal

A = 0.007 *0.004 
k = 120 #watt pro meter kelvin
dQ_100 = A*k*((T3[20]-T4[20])/0.03)
dQ_200 = A*k*((T3[40]-T4[40])/0.03)
dQ_350 = A*k*((T3[70]-T4[70])/0.03)
dQ_450 = A*k*((T3[90]-T4[90])/0.03)
dQ_600 = A*k*((T3[120]-T4[120])/0.03)
print('dQ:', dQ_100, dQ_200, dQ_350, dQ_450, dQ_600)

#aluminium

A = 0.012 *0.004 
k = 237 #watt pro meter kelvin
dQ_100 = -A*k*((T5[20]-T6[20])/0.03)
dQ_200 = -A*k*((T5[40]-T6[40])/0.03)
dQ_350 = -A*k*((T5[70]-T6[70])/0.03)
dQ_450 = -A*k*((T5[90]-T6[90])/0.03)
dQ_600 = -A*k*((T5[120]-T6[120])/0.03)
print('dQ:', dQ_100, dQ_200, dQ_350, dQ_450, dQ_600)

#edelstahl

A = 0.012 *0.004 
k = 15 #watt pro milli kelvin
dQ_100 = A*k*((T7[20]-T8[20])/0.03)
dQ_200 = A*k*((T7[40]-T8[40])/0.03)
dQ_350 = A*k*((T7[70]-T8[70])/0.03)
dQ_450 = A*k*((T7[90]-T8[90])/0.03)
dQ_600 = A*k*((T7[120]-T8[120])/0.03)
print('dQ:', dQ_100, dQ_200, dQ_350, dQ_450, dQ_600)

##dynamische methode
#messing
A_fern = np.array([35.45-24.52, 42.91-34.90, 48.28-41.51, 52.66-46.23, 56.24-50.86, 58.82-53.53, 61.08-55.97, 62.97-58.05, 64.75-59.90, 66.17-61.52, 67.45-62.86])
A_nah = np.array([45.62-25.81, 52.36-36.29, 57.26-42.09, 60.84-46.26, 64.68-51.13, 67.22-53.24, 69.45-55.69, 71.25-57.70, 73.00-59.54, 74.45-61.18, 75.75-62.49])
print('A_fern ', A_fern)
print('A_nah ', A_nah)
D = np.log(A_nah/A_fern)
print('ln() für Messing', D)
print( np.mean(D), np.std(D)/11)

dt = np.array([22, 16, 16, 20, 14, 12, 12, 12, 12, 12, 12])
print('dt', np.mean(dt), np.std(dt)/11)

k = (8520*385*(0.03**2))/(2*dt*D)
print('k', k)
print('k=', np.mean(k), np.std(k)/11)

lam = 2*np.pi / np.sqrt((2*np.pi*dt*D)/(80* 0.03**2))
print('lambda = ', np.mean(lam), '+- ', np.std(lam)/11)

#aluminium
A_fern = np.array([41.68-24.77, 50.07-38.58, 55.14-45.13, 58.88-49.29, 62.47-54.05, 64.67-56.01, 66.76-58.25, 68.43-60.15, 70.08-61.85, 71.43-63.35, 72.71-64.65])
A_nah = np.array([49.70-25.74, 57.34-38.15, 62.06-44.10, 65.34-48.07, 69.11-52.86, 71.33-54.62, 73.35-56.92, 75.11-58.88, 76.79-60.57, 78.07-62.06, 79.38-63.48])
print('A_fern ', A_fern)
print('A_nah ', A_nah)
D = np.log(A_nah/A_fern)
print('ln() für Aluminium', D)
print( np.mean(D), np.std(D)/11)

dt = np.array([12, 8, 8, 14, 6, 8, 6, 6, 6, 8, 6])
print('dt', np.mean(dt), np.std(dt)/11)

k = (2800*830*(0.03**2))/(2*dt*D)
print('k', k)
print('k=', np.mean(k), np.std(k)/11)

lam = 2*np.pi / np.sqrt((2*np.pi*dt*D)/(80* 0.03**2))
print('lambda = ', np.mean(lam), '+- ', np.std(lam)/11)


#edelstahl
A_fern = np.array([34.41-26.39, 40.02-34.17, 44.40-39.39, 47.84-43.42])
A_nah = np.array([57.19-31.22, 63.31-40.73, 67.77-46.05, 71.12-50.05])
print('A_fern ', A_fern)
print('A_nah ', A_nah)
D = np.log(A_nah/A_fern)
print('ln() für Edelstahl', D)
print( np.mean(D), np.std(D)/4)

dt = np.array([76, 66, 62, 56])
print('dt', np.mean(dt), np.std(dt)/4)

k = (8000*400*(0.03**2))/(2*dt*D)
print('k', k)
print('k=', np.mean(k), np.std(k)/4)

lam = 2*np.pi / np.sqrt((2*np.pi*dt*D)/(80* 0.03**2))
print('lambda = ', np.mean(lam), '+- ', np.std(lam)/4)