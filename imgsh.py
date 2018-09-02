import cv2
import numpy as np
import os
# cam = cv2.VideoCapture(0)
# c = 1;
# while c<5500:
# _,img=cam.read()
def imgcnv(string):
    img=cv2.imread(string)
    img=cv2.resize(img,(512, 512))
    # cv2.putText(img,'no. of finger: 1',(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0))
    cv2.imshow('img',img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    op = cv2.bitwise_and(img, img, mask=mask)

    # _, contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(op, contour, -1, (0, 255, 0), thickness=3)

    # cv2.imshow("HSV", hsv)
    # cv2.imshow("mask", mask)
    # cv2.imwrite('mask.png', mask)
    # c = c + 1
    # sname = str(c) + '.jpg'

    cv2.imshow('g', op)
    cv2.imwrite('12hsv.jpg',op)
    # print(c)
    while True:
        k= cv2.waitKey(10)
        if k==27:
            break
    # cam.release()
    cv2.destroyAllWindows()

imgcnv('12.jpg')