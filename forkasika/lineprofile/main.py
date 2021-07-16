import numpy as np
import cv2 

HEIGHT = 1024
WIDTH = 1024
WAVE_LEN = 0.6328
DX = 10.0
POSI_Z = 50000.0
DIAM = 50.0
PEAK_BRIGHT = 127

DIST = 50.0

def trans_func(posi_z):
    z = np.full([HEIGHT,WIDTH],0.+0.j)
    for i in range(HEIGHT):
        for k in range(WIDTH):
            tmp = 2.0j*np.pi*posi_z/WAVE_LEN*np.sqrt(1.0-((k - WIDTH/2)*WAVE_LEN/WIDTH/DX)**2 -((i - HEIGHT/2)*WAVE_LEN/HEIGHT/DX)**2 )
            z[i,k] = np.exp(tmp)
    return z

def object (posi_x,posi_y):
    z = np.full([HEIGHT,WIDTH],0.+0.j)
    for i in range(HEIGHT):
        for k in range(WIDTH):
            if (i*DX-posi_y)**2 + (k*DX-posi_x)**2 > (DIAM/2)**2 :
                z[i,k] = 1.0+0.j
    return z

trans1 = trans_func(POSI_Z)

posi_x1 = WIDTH*DX/2.0
posi_y1 = HEIGHT*DX/2.0

object1 = object(posi_x1, posi_y1)

object_plane = object1

objectfft = np.fft.fft2(object_plane)
objectfft = np.fft.fftshift(objectfft)

hologram = objectfft*trans1
hologram = np.fft.fftshift(hologram)
hologram = np.fft.ifft2(hologram)
rehologram = PEAK_BRIGHT* np.abs(hologram)
cv2.imwrite("holo.png",rehologram)

array = np.abs(hologram[512:,512])
np.savetxt("lineprofile.txt",array)