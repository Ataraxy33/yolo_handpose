# -*- coding: utf-8 -*-
import cv2
 
cap = cv2.VideoCapture(1) # 0表示第一个摄像头
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()