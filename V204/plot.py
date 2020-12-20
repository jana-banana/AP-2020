import matplotlib.pyplot as plt
import numpy as np

t, T1, T4, T5, T8, T2, T3, T6, T7 = np.genfromtxt('data_stat.txt', unpack=True)

plt.plot(5*t, T1, 'b.', label='Messing (breit) - T1')
plt.plot(5*t, T4, 'r.', label='Messing (schmal) - T4')

plt.plot(5*t, T5, 'k.', label='Aluminium - T5')
plt.plot(5*t, T8, 'g.', label='Edelstahl - T8')

plt.xlabel(r'$t \mathbin{/} \si{\second} $')
plt.ylabel(r'$T \mathbin{/} \si{\celsius}$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('build/stat_plot.pdf')

plt.clf()
DT78 = T7-T8
DT21 = T2-T1
plt.plot(5*t, DT78, 'k.', label='Temperaturdifferenz Edelstahl')
plt.plot(5*t, DT21, 'r.', label='Temperaturdifferenz Messing (breit)')

plt.xlabel(r'$t \mathbin{/} \si{\second}$')
plt.ylabel(r'$T \mathbin{/} \si{\celsius}$')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/plot_diff.pdf')

plt.clf()
##dynamische Methode
t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('data_dyn_40.txt', unpack=True)
#messingstab-breit
plt.plot(2*t, T1, 'k.', label=r'$T_{\text{fern}}$')
plt.plot(2*t, T2, 'r.', label=r'$T_{\text{nah}}$')

plt.xlabel(r'$t \mathbin{/} \si{\second} $')
plt.ylabel(r'$T \mathbin{/} \si{\celsius}$')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/plot_messing.pdf')

plt.clf()
#aluminium
plt.plot(2*t, T5, 'k.', label=r'$T_{\text{fern}}$')
plt.plot(2*t, T6, 'r.',label=r'$T_{\text{nah}}$')

plt.xlabel(r'$t \mathbin{/} \si{\second} $')
plt.ylabel(r'$T \mathbin{/} \si{\celsius}$')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/plot_aluminium.pdf')

plt.clf()
#edelstahl
t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('data_dyn_100.txt', unpack=True)
plt.plot(2*t, T7, 'r.', label=r'$T_{\text{nah}}$')
plt.plot(2*t, T8, 'k.', label=r'$T_{\text{fern}}$')

plt.xlabel(r'$t \mathbin{/} \si{\second} $')
plt.ylabel(r'$T \mathbin{/} \si{\celsius}$')

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/plot_edelstahl.pdf')