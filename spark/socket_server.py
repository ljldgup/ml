# ！/usr/bin/env python
# coding:utf-8
# 导入socket模块
import socket

word = ['cat', 'dog', 'you']
# 开启ip和端口
ip_port = ('127.0.0.1', 9999)
# 生成句柄
web = socket.socket()
# 绑定端口
web.bind(ip_port)
# 最多连接数
web.listen(5)
# 等待信息
print('waiting...')
# 开启死循环
while True:
    # 阻塞
    conn, addr = web.accept()
    print('connect from', addr)
    # 获取客户端请求数据
    # data = conn.recv(1024)
    # 打印接受数据 注：当浏览器访问的时候，接受的数据的浏览器的信息等。
    # print(data)
    # 向对方发送数据
    while 1:
        input_word = input()
        conn.send(bytes(input_word, 'utf-8'))
    # 关闭链接
    conn.close()
