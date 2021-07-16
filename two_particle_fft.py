import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

HEIGHT = 512
WIDTH = 512
C_TRAIN_COUNT = 3000
F_TRAIN_COUNT = 3000
C_TEST_COUNT = 500
F_TEST_COUNT = 500

close_train_images = glob.glob("./datasets/two_particle_datasets_close/*")
far_train_images = glob.glob("./datasets/two_particle_datasets_far/*")
close_test_images = glob.glob("./datasets/two_particle_datasets_close_test/*")
far_test_images = glob.glob("./datasets/two_particle_datasets_far_test/*")

close_train_images.sort()
far_train_images.sort()
close_test_images.sort()
far_test_images.sort()

for i in range(C_TRAIN_COUNT):
    data = cv2.imread(close_train_images[i],0)
    fft = np.fft.fft2(data)
    fft = np.fft.fftshift(fft)
    fft = np.abs(fft)
    fft = np.log(fft+1)
    image = 255*(fft - np.min(fft))/(np.max(fft) - np.min(fft))
    cv2.imwrite(f"./datasets/fft_two_particle_datasets_close/{i:05}.png",image)

for i in range(F_TRAIN_COUNT):
    data = cv2.imread(far_train_images[i],0)
    fft = np.fft.fft2(data)
    fft = np.fft.fftshift(fft)
    fft = np.abs(fft)
    fft = np.log(fft+1)
    image = 255*(fft - np.min(fft))/(np.max(fft) - np.min(fft))
    cv2.imwrite(f"./datasets/fft_two_particle_datasets_far/{i:05}.png",image)

for i in range(C_TEST_COUNT):
    data = cv2.imread(close_test_images[i],0)
    fft = np.fft.fft2(data)
    fft = np.fft.fftshift(fft)
    fft = np.abs(fft)
    fft = np.log(fft+1)
    image = 255*(fft - np.min(fft))/(np.max(fft) - np.min(fft))
    cv2.imwrite(f"./datasets/fft_two_particle_datasets_close_test/{i:05}.png",image)

for i in range(F_TEST_COUNT):
    data = cv2.imread(far_test_images[i],0)
    fft = np.fft.fft2(data)
    fft = np.fft.fftshift(fft)
    fft = np.abs(fft)
    fft = np.log(fft+1)
    image = 255*(fft - np.min(fft))/(np.max(fft) - np.min(fft))
    cv2.imwrite(f"./datasets/fft_two_particle_datasets_far_test/{i:05}.png",image)

    

