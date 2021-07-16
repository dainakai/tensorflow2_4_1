import numpy as np
import cv2

HEIGHT = 512
WIDTH = 512
WAVE_LEN = 0.6328
DX = 10.0
PEAK_BRIGHT = 127
PHASE_DIST = 50000
HOLO_DIST = 50000
PHASE_ITR = 20

imagepath1 = "./holo1.png"
image1 = cv2.imread(imagepath1,0)
imagepath2 = "./holo2.png"
image2 = cv2.imread(imagepath2,0)

def trans_func(posi_z):
    z = np.full([HEIGHT,WIDTH],0.+0.j)
    for i in range(HEIGHT):
        for k in range(WIDTH):
            tmp = 2.0j*np.pi*posi_z/WAVE_LEN*np.sqrt(1.0-((k - WIDTH/2)*WAVE_LEN/WIDTH/DX)**2 -((i - HEIGHT/2)*WAVE_LEN/HEIGHT/DX)**2 )
            z[i,k] = np.exp(tmp)
    return z

intensity1 = image1/PEAK_BRIGHT
intensity2 = image2/PEAK_BRIGHT

trans1 = trans_func(PHASE_DIST)
trans2 = trans_func(-PHASE_DIST)
trans3 = trans_func(-HOLO_DIST)

holo = np.fft.fft2(intensity1)
holo = np.fft.fftshift(holo)
holo = holo*trans3
holo = np.fft.fftshift(holo)
holo = np.fft.ifft2(holo)
power = PEAK_BRIGHT*np.abs(holo)
power = power.astype('int64')
cv2.imwrite("Gaborholo.png",power)

phi1 = np.full([HEIGHT,WIDTH], 0.+0.j)
phi1 = 0.j + intensity1
phi2 = np.full([HEIGHT,WIDTH], 0.+0.j)
phi2 = 0.j + intensity2

for itr in range(PHASE_ITR):
    phi1fft = np.fft.fft2(phi1)
    phi1fft = np.fft.fftshift(phi1fft)

    phi2fft = phi1fft * trans1

    phi2fft = np.fft.fftshift(phi2fft)
    phi2 = np.fft.ifft2(phi2fft)

    theta2 = np.arctan2(np.imag(phi2), np.real(phi2))
    phi2 = intensity2*np.exp(1.j*theta2)

    phi2fft = np.fft.fft2(phi2)
    phi2fft = np.fft.fftshift(phi2fft)

    phi1fft = phi2fft* trans2

    phi1fft = np.fft.fftshift(phi1fft)
    phi1 = np.fft.ifft2(phi1fft)

    theta1 = np.arctan2(np.imag(phi1), np.real(phi1))

    phi1 = intensity1*np.exp(1.j*theta1)

phi1 = np.fft.fft2(phi1)
phi1 = np.fft.fftshift(phi1)

holo = phi1 * trans3

holo = np.fft.fftshift(holo)
holo = np.fft.ifft2(holo)

power = PEAK_BRIGHT*np.abs(holo)
power = power.astype('int64')

cv2.imwrite("phaseretrieved.png", power)