import numpy as np
import glob
import cv2
import os

RADUIS_BUF = 20
CROP_LENGTH = 256
BRIGHT = 127
IMAGE_LEN = 1024

datapaths = glob.glob("./data/*")

for path in datapaths:
    data = np.loadtxt(path)
    input_image = cv2.imread(path.replace("./data", "./images").replace("_hough.txt", ".bmp"),0)
    os.makedirs(path.replace("./data", "./cropimage").replace("_hough.txt", ""), exist_ok=True)

    for idx in range(len(data[:,0])):
        x_posi = data[idx,0]
        y_posi = data[idx,1]
        radius = data[idx,2]+RADUIS_BUF
        if radius/2 != radius//2:
            radius +=1

        crop = input_image[max(int(y_posi-radius),0):min(int(y_posi+radius),IMAGE_LEN), max(int(x_posi-radius),0):min(int(x_posi+radius),IMAGE_LEN)]

        new_image = np.full((CROP_LENGTH,CROP_LENGTH),BRIGHT)

        width = crop.shape[1]
        height = crop.shape[0]

        new_image[CROP_LENGTH//2 - height//2:CROP_LENGTH//2 + height//2, CROP_LENGTH//2 - width//2:CROP_LENGTH//2 + width//2] = crop[:,:]

        cv2.imwrite(path.replace("./data", "./cropimage").replace("_hough.txt", "") + f"/{idx:03}.png", new_image)


