import cv2
import socket
import pickle
import struct

# 创建 TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.1.2', 8089)  # 替换成你电脑的IP地址

# 连接服务器
client_socket.connect(server_address)

# 打开摄像头
camera = cv2.VideoCapture(1)  # 默认摄像头

try:
    while True:
        ret, frame = camera.read()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(encoded_frame, protocol=pickle.HIGHEST_PROTOCOL)

        # 发送图像数据
        client_socket.sendall(struct.pack('>L', len(data)) + data)

        

finally:
    client_socket.close()
    camera.release()
