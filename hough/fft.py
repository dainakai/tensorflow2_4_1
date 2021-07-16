import numpy as np
import glob
import cv2
import os

paths = glob.glob("./cropimage/*/*")

for path in paths:
    os.makedirs(path[:-7].replace("./cropimage", "./crop_fft"), exist_ok=True)
    image = cv2.imread(path,0)
    image = np.fft.fft2(image)
    image = np.fft.fftshift(image)
    image = np.abs(image)
    image = np.log(image + 1)
    data = 255*(image - np.min(image))/(np.max(image) - np.min(image))
    image_path = path.replace("./cropimage", "./crop_fft")
    cv2.imwrite(image_path,data)

