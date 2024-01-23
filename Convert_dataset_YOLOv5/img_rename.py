import os
import cv2

root = r"D:\Python\source\Mouse_Keyboard\img"
save_path = r"D:\Python\source\Mouse_Keyboard\ori_img"
for i, ori_name in enumerate(os.listdir(root)):
    print(ori_name)
    img = cv2.imread(os.path.join(root, ori_name))
    new_name = str(0) + str(i + 401) + ".jpg"
    cv2.imwrite(os.path.join(save_path, new_name), img)
