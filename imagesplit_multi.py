import numpy as np
import glob
import os
import cv2
import sys
from multiprocessing import Pool

def fft (input):
    data = np.fft.fft2(input)
    data = np.fft.fftshift(data)
    data = np.abs(data)
    data = np.log(data+1)
    image = 255*(data - np.min(data))/(np.max(data) - np.min(data))
    return image.astype(np.uint8)

def splitter (input, num, opt, height = 1024):
    '''
        Image data length needs to be divisible by 2*num.
        Be careful!!!!
    '''
    k1 = np.arange(0,num,1)
    k2 = np.arange(0,num-1,1)

    column1 = height*k1//num
    column1 = column1.astype(np.int16)
    column2 = height*(k2/num + 0.5/num)
    column2 = column2.astype(np.int16)

    x0 = column1
    y0 = column1

    x1 = column2
    y1 = column1

    x2 = column1
    y2 = column2

    x3 = column2
    y3 = column2


    count = num*num
    output0 = np.ndarray((height//num,height//num*count))

    for y in range(num):
        for x in range(num):
            output0[:,height//num*(y*num+x):height//num*(y*num+x+1)] = fft(input[y0[y]:y0[y]+height//num, x0[x]:x0[x]+height//num])

    if opt == 0:
        return output0

    count = num*(num-1)
    output1 = np.ndarray((height//num,height//num*count))
    output2 = np.ndarray((height//num,height//num*count))

    for y in range(num):
        for x in range(num-1):
            output1[:,height//num*(y*(num-1)+x):height//num*(y*(num-1)+x+1)] = fft(input[y1[y]:y1[y]+height//num, x1[x]:x1[x]+height//num])

    for y in range(num-1):
        for x in range(num):
            output2[:,height//num*(y*num+x):height//num*(y*num+x+1)] = fft(input[y2[y]:y2[y]+height//num, x2[x]:x2[x]+height//num])

    output = np.concatenate([output0,output1,output2],1)

    if opt == 1:
        return output

    count = (num-1)*(num-1)
    output3 = np.ndarray((height//num,height//num*count))

    for y in range(num-1):
        for x in range(num-1):
            output3[:,height//num*(y*num+x):height//num*(y*num+x+1)] = fft(input[y3[y]:y3[y]+height//num, x3[x]:x3[x]+height//num])

    output = np.concatenate([output,output3],1)
    
    if opt == 2:
        return output

def imageProcess(path):
    image = cv2.imread(path,0)
    output = splitter(image, num, opt)
    output_path = path.replace(IMAGE_DIR,OUTPUT_DIR).replace(".bmp",".png")
    cv2.imwrite(output_path,output)
    print(path.replace(IMAGE_DIR,"")+" th have been processed.")

args = sys.argv
traintest = args[1]
closefar = args[2]
opt = int(args[3])
num = int(args[4])
pnum = int(args[5])

IMAGE_DIR = f"/media/dai/DATADISK/two_particle_datasets/holograms/{traintest}/{closefar}_holo/num_{pnum:05}/"
OUTPUT_DIR = f"/media/dai/DATADISK/two_particle_datasets/segmentation/{traintest}/{closefar}_holo/opt{opt}/segnum{num}/num_{pnum:05}/"
paths = glob.glob(f"{IMAGE_DIR}*")
os.makedirs(OUTPUT_DIR, exist_ok=True)

p = Pool(24)
p.map(imageProcess,paths)
