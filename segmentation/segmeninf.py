from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import sys

args = sys.argv
opt = int(args[1])
segnum = int(args[2])
pnum = int(args[3])
testpnum = int(args[4])

epoch_num = 150
if opt == 0:
    count = segnum*segnum
    bs = 8
elif opt == 1:
    count = segnum*segnum + 2*segnum*(segnum-1)
    bs = 8
elif opt == 2:
    count = segnum*segnum + 2*segnum*(segnum-1) + (segnum-1)*(segnum-1)
    bs = 4

HEIGHT = 128
WIDTH = 128*count
MODEL_PATH = f"./segmodels/opt{opt}/segnum{segnum}/num_{pnum:05}/"

close_test_path = glob.glob(f"/media/dai/DATADISK/two_particle_datasets/segmentation/test/close_holo/opt{opt}/segnum{segnum}/num_{(testpnum-1):05}/*")
far_test_path = glob.glob(f"/media/dai/DATADISK/two_particle_datasets/segmentation/test/far_holo/opt{opt}/segnum{segnum}/num_{testpnum:05}/*")

print(len(close_test_path))
print(len(far_test_path))

#IMAGE ARRAY INITIALIZING

test_image = []
test_label = []

#INPUTTING IMG TO ARRAY
for ii in close_test_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    test_image.append(img)
    test_label.append(1)

for ii in far_test_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    test_image.append(img)
    test_label.append(0)

test_image = np.asarray(test_image)
test_label = np.asarray(test_label)

#NORMALIZATION, ARRAY CASTING FOR GPU CALCULATION
test_image = test_image.astype('float32')
test_image = test_image / 255.0

#DATASETS SHUFFLING
for ii in [test_image, test_label]:
    np.random.seed(1)
    np.random.shuffle(ii)

model = models.load_model(MODEL_PATH)
#EVALUATION WITH TEST DATA
test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)

print("test data loss :",test_loss)
print("test data acc :",test_acc)

file = open(f"./accdata/acc_opt{opt}_seg{segnum}_num{pnum}.txt","a")
file.write(f"{testpnum}\t{test_acc}\n")
file.close()
