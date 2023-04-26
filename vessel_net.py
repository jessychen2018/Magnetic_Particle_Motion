import numpy as np

def vessel_net_3d(file_dir):
    file = open(file_dir, "r", encoding='utf-8')
    lists = file.readlines()
    initial_row = 10
    lists_ = []
    j = 0
    while 1:
        if lists[initial_row + j].split()[0] == '%':
            break
        lists_.append(lists[initial_row + j].split())
        j = j + 1
    lists = np.array(lists_)
    lists = lists.astype(np.float32)+20

    x = lists[:, 0]
    ma1 = int(max(x) + 5)  # 找出x坐标最大值
    y = lists[:, 1]
    ma2 = int(max(y) + 10)  # 找出y坐标最大值
    z = lists[:, 2]
    ma3 = int(max(z) +40)  # 找出z坐标最大值
    web = np.empty((ma1, ma2, ma3))

    for k in range(len(lists[:, 0])):
        u = round(lists[k][0])
        v = round(lists[k][1]) + 5
        w = round(lists[k][2]) + 20
        web[u, v, w] = 255

    ct = np.where(web == 255)
    ct = np.array(ct)
    for k in range(len(ct[0])):
        u = ct[0][k]
        v = ct[1][k]
        w = ct[2][k]
        if web[u - 1:u + 2, v - 1:v + 2, w - 1:w + 2].all() == 0:
            web[u - 1:u + 2, v - 1:v + 2, w - 1:w + 2][web[u - 1:u + 2, v - 1:v + 2, w - 1:w + 2] == 0] = 100
    return web, ma1, ma2, ma3


