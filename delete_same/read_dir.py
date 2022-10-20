import os
from natsort import ns, natsorted

path = '/home/room/PycharmProjects/yolov5s_fire/runs/detect/gas_train/vidcut'
# for file in sorted(os.listdir(path)):
#     print(file)

file_list = os.listdir(path)
files = natsorted(file_list, alg=ns.PATH)
print(files)
