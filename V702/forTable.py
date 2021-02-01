import numpy as np

f= open("vanatab.txt","w+")

t ,Nmess, N, dN = [],[], [], []
for line in open('vanadium.txt', 'r'):
  values = [float(s) for s in line.split()]
  t.append(values[0])
  Nmess.append(values[1])
  N.append(values[1]-13.9)
  dN.append(np.sqrt(values[1]-13.9))

for i in range(36):
    a = str(t[i])+" & "+str(Nmess[i])+" & "+str(N[i])+" & "+str(dN[i])
    a += "\ "
    a += "\ "
    f.write(a)
    f.write("\n")

f.close()