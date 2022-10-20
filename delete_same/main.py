import os
import shutil
import numpy as np
import cv2
from natsort import ns, natsorted


# 计算两张图片的相似度
def calc_similarity(img1_path, img2_path):
    img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), -1)
    H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])  # 计算图直方图
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
    img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), -1)
    H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])  # 计算图直方图
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
    similarity1 = cv2.compareHist(H1, H2, 0)  # 相似度比较
    print('similarity:', similarity1)
    if similarity1 > 0.99:  # 0.93是阈值，可根据需求调整
        return True
    else:
        return False


# 去除相似度高的图片

def filter_similar(dir_path):
    filter_dir = os.path.join(os.path.dirname(dir_path), dir_path.split('/')[-1] + '_' + 'similar')
    # print(os.path.dirname(dir_path))
    # print(dir_path.split('/')[-1])
    # print(filter_dir)
    # exit()
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    loop_num = 0
    while True:
        loop_num += 1
        filter_number = 0
        for root, dirs, files in os.walk(dir_path):
            img_files = [file_name for file_name in files]
            # img_files = sorted(img_files)
            img_files = natsorted(img_files, alg=ns.PATH)
            # print(img_files)
            # exit()
            filter_list = []
            if len(img_files) >= 5:
                for index in range(len(img_files))[:-4]:
                    if img_files[index] in filter_list:
                        continue
                    for idx in range(len(img_files))[(index + 1):(index + 5)]:
                        img1_path = os.path.join(root, img_files[index])
                        img2_path = os.path.join(root, img_files[idx])
                        if calc_similarity(img1_path, img2_path) and img_files[idx] not in filter_list:
                            filter_list.append(img_files[idx])
                            filter_number += 1
                for item in filter_list:
                    src_path = os.path.join(root, item)
                    shutil.move(src_path, filter_dir)
            else:
                print("num of pic < 5 !")
                break
        if filter_number == 0:
            print("第{}次执行".format(loop_num))
            print(filter_number)
            break
        else:
            print("第{}次执行".format(loop_num))
            print(filter_number)
    return loop_num


if __name__ == '__main__':
    path0 = '/home/room/TEST_JYF/key_frame'
    path1 = '/home/room/TEST_JYF/key_frame_smoke'
    path2 = '/home/room/source/yolodata/fire/key_frame'
    path3 = '/home/room/source/yolodata/smoke/key_frame_cd'
    path4 = '/home/room/PycharmProjects/yolov5s_fire/runs/detect/gas_train/vidcut'
    path5 = '/home/room/PycharmProjects/yolov5s_fire/runs/detect/gas0/vidcut'
    path6 = '/home/room/PycharmProjects/yolov5s_fire/runs/detect/gas1/vidcut'
    path7 = '/home/room/PycharmProjects/yolov5s_smoke/runs/detect/gas_train/vidcut'
    path8 = '/home/room/PycharmProjects/yolov5s_smoke/runs/detect/gas0/vidcut'
    path9 = '/home/room/PycharmProjects/yolov5s_smoke/runs/detect/exp_positive/vidcut'
    # print(calc_similarity('/home/room/PycharmProjects/yolov5s_fire/runs/detect/gas_train/vidcut/4_0_1_14_0_1.jpg',
    #                       '/home/room/PycharmProjects/yolov5s_fire/runs/detect/gas_train/vidcut/4_0_1_24_0_1.jpg'))
    path_list = [path9]
    for i in range(1):
        print(path_list[i])
        print(filter_similar(path_list[i]))
