import os
import cv2

dlist = os.listdir('spred')
c=0
print(dlist)

for i in dlist:
    string = 'spred\\' + i
    img = cv2.imread(string,0)
    string = 'spred\\' + i
    cv2.imwrite(string, img)
    print(i)
    c=c+1

print('done')