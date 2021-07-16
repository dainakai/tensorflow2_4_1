import numpy as np

array = np.loadtxt("./weights.txt")
t0 = array[:,0]
t1 = array[:,1]

t0_array = np.zeros([62,62])
t1_array = np.zeros([62,62])
for i in range(62):
    for j in range(62):
        t0_array[i,j] = t0[i*62 + j]
        t1_array[i,j] = t1[i*62 + j]

np.savetxt("./t0weights.txt",t0_array)
np.savetxt("./t1weights.txt",t1_array)