import numpy as np
import cv2

image = cv2.imread("./raw/11_close.bmp",0)
cv2.imwrite("./raw/11_close.png",image)

data = np.fft.fft2(image)
data = np.fft.fftshift(data)
data = np.abs(data)
data = np.log(data+1)
image = 255*(data - np.min(data))/(np.max(data) - np.min(data))

cv2.imwrite("./raw/11_close_fft.png",image)