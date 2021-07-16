import numpy as np
import cv2
import random

HEIGHT = 512
WIDTH = 512
WAVE_LEN = 0.6328
DX = 10.0
DIAM = 80.0
PARTICLE_DIST = 120.0
HOLO_DIST = 50000
PEAK_BRIGHT = 127

def trans_func(posi_z):
    z = np.full([HEIGHT,WIDTH],0.+0.j)
    for i in range(HEIGHT):
        for k in range(WIDTH):
            tmp = 2.0j*np.pi*posi_z/WAVE_LEN*np.sqrt(1.0-((k - WIDTH/2)*WAVE_LEN/WIDTH/DX)**2 -((i - HEIGHT/2)*WAVE_LEN/HEIGHT/DX)**2 )
            z[i,k] = np.exp(tmp)
    return z

drop1 = [WIDTH/2*random.random()+WIDTH/4,HEIGHT/2*random.random()+HEIGHT/4]
theta = 2.0 * np.pi * random.random() 
drop2 = [drop1[0] + PARTICLE_DIST/DX*np.cos(theta),drop1[1] + PARTICLE_DIST/DX*np.sin(theta)]

object = np.zeros([HEIGHT,WIDTH])
for i in range(HEIGHT):
    for k in range(WIDTH):
        if (k-drop1[0])**2 + (i - drop1[1])**2 > (DIAM/DX/2)**2 and (k-drop2[0])**2 + (i - drop2[1])**2 > (DIAM/DX/2)**2 :
            object[k,i] = 1.0

trans1 = trans_func(HOLO_DIST)
trans2 = trans_func(HOLO_DIST*2)

object = np.fft.fft2(object)
object = np.fft.fftshift(object)

holo1 = object*trans1
holo2 = object*trans2

holo1 = np.fft.fftshift(holo1)
holo1 = np.fft.ifft2(holo1)
holo2 = np.fft.fftshift(holo2)
holo2 = np.fft.ifft2(holo2)

power1 = PEAK_BRIGHT * np.abs(holo1)
power1 = power1.astype('int64')

cv2.imwrite("holo1.png", power1)

power2 = PEAK_BRIGHT * np.abs(holo2)
power2 = power2.astype('int64')

cv2.imwrite("holo2.png", power2)