import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
#     img= cv2.imread('ct2.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0,48,80],dtype='uint8')
    upper = np. array([20,255,255], dtype='uint8')

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask= cv2.erode( mask, kernel,iterations=2)
    mask=cv2.dilate(mask, kernel,iterations=2)

    mask = cv2.GaussianBlur(mask, (3,3), 0)

    op= cv2.bitwise_and(img, img, mask=mask)

#     _, contour ,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(op, contour, -1, (0,255,0), thickness=3)
#
#     # cv2.imshow("HSV", hsv)
#     # cv2.imshow("mask", mask)
# # cv2.imwrite('mask.png',mask)
# # cv2.imwrite('op.png',op)
#
    cv2.imshow("God Hand", op)

    k= cv2.waitKey(30)
    if k==27:
        break


cam.release()
cv2.destroyAllWindows()