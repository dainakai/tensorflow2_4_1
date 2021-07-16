import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt

WAVE_LEN = 0.6328
DX = 10.0
POSI_Z = 50000.0
DIAM = 50.0
PEAK_BRIGHT = 127

def func(x):
    y = np.zeros(len(x))
    c1 = np.pi*(DIAM/2)**2/WAVE_LEN/POSI_Z
    c2 = np.pi/WAVE_LEN/POSI_Z
    c3 = 2.0*np.pi*(DIAM/2)/WAVE_LEN/POSI_Z
    
    for idx in range(len(x)):
        if x[idx] != 0:
            y[idx] = 1 - 2.0*c1*np.sin(c2*x[idx]*x[idx])*(2.0*j1(c3*x[idx]))/(c3*x[idx]) + c1**2*((2.0*j1(c3*x[idx]))/(c3*x[idx]))**2
        else:
            y[idx] = 1
    return y

x = np.arange(51200)/10
y = func(x)

np.savetxt("theory.txt",y)

# plt.plot(x,y)
# plt.legend()
# plt.show()

