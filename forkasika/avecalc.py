import numpy as np

array = np.loadtxt("twoaccandloss.txt")

acc = np.sum(array[:,1])/10
loss = np.sum(array[:,0])/10

print(f"acc:{acc} loss:{loss}\n")