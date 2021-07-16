import cv2
import numpy as np
import glob

paths = glob.glob("./images/*.bmp")

def houghCircle (path):
    img = cv2.imread(path,0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1.0,50,
                                param1=20.495,param2=30,minRadius=0,maxRadius=100)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    if ".png" in path :
        path = path.replace(".png","_hough.png").replace("./images", "./results")
    elif ".bmp" in path :
        path = path.replace(".bmp","_hough.png").replace("./images", "./results")

    cv2.imwrite(path,cimg)

    np.savetxt(path.replace(".png",".txt").replace("./results", "./data"), circles.reshape(-1,3))

for idx in paths:
    houghCircle(idx)