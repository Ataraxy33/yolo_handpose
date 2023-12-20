import cv2
import socket
import numpy as np
import pickle

UDP_IP = '0.0.0.0'
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

cv2.namedWindow('Raspberry Pi Camera', cv2.WINDOW_NORMAL)

try:
    while True:
        # 接收画面数据
        image, addr = sock.recvfrom(100000)
        image = pickle.loads(image)
        image = np.frombuffer(image, dtype='uint8')
        image = cv2.imdecode(image, 1)
        cv2.imshow('Raspberry Pi Camera', image)

        # 检测按键，按下Esc键退出程序
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
finally:
    # 关闭窗口和套接字
    cv2.destroyAllWindows()
    sock.close()
