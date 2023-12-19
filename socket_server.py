import cv2
import socket
import pickle
import struct

# 创建 TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'localhost'  # 监听所有网络接口
port = 8089
server_socket.bind((host, port))
server_socket.listen(1)

# 接受连接
connection, client_address = server_socket.accept()
print('Connection from', client_address)

# 设置窗口大小
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 640, 480)

try:
    while True:
        # 接收图像大小信息
        data_size = struct.unpack('>L', connection.recv(struct.calcsize('>L')))[0]

        # 接收并解析图像数据
        data = b''
        while len(data) < data_size:
            packet = connection.recv(data_size - len(data))
            if not packet:
                break
            data += packet

        # 解pickle并显示图像
        frame = pickle.loads(data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Frame', frame)

        # 检测按键，按下 Esc 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            # 发送退出信号，用 1 表示退出
            break
finally:
    connection.close()
    server_socket.close()
    cv2.destroyAllWindows()