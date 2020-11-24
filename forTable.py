#A 
#B 
#C
#D
#E
#K1
#K2

f= open("hope.txt","w+")

Z,T1,p1,T2,p2,N,K1,K2= [], [], [],[],[],[],[],[]
for line in open('data.txt', 'r'):
  values = [float(s) for s in line.split()]
  Z.append(values[0])
  T1.append(values[1])
  K1.append(values[1]+273.15)
  p1.append(values[2]+1)
  T2.append(values[3])
  K2.append(values[4]+273.15)
  p2.append(values[4]+1)
  N.append(values[5])

for i in range(36):
    a = str(Z[i])+" & "+str(T1[i])+" & "+str(K1[i])+" & "+str(p1[i])+" & "+str(T2[i])+" & "+str(K1[i])+" & "+str(p2[i])+" & "+str(N[i])
    a += "\ "
    a+= " \ "
    f.write(a)
    f.write("\n")

f.close()