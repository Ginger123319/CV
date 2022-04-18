import os
from PIL import Image, UnidentifiedImageError
import numpy as np

bg_path = "../../source/target_detection/test_pic_original"
x = 401
for filename in os.listdir(bg_path)[::-1]:
    print(filename)
    # print(f"x is {x}")
    # new_name = filename.split(".")
    # print(new_name)
    # tag = f"{x}"
    # print(x)
    # new_name[0] = tag
    # print(new_name)
    # # 有pycharm显示不完整的情况，并非程序有错误，建议建个小的测试demo【含几个文件的文件夹目录】测试代码
    # os.rename(os.path.join(bg_path, filename), os.path.join(bg_path, ".".join(new_name)))
    # # print(filename)

    try:
        background = Image.open("{0}/{1}".format(bg_path, filename))
    except UnidentifiedImageError:
        continue
    else:
        shape = np.shape(background)
        if len(shape) == 3 and shape[0] > 100 and shape[1] > 100:
            background = background
        else:
            continue
        background_resize = background.resize((300, 300))
        background_resize = background_resize.convert("RGB")
        name = np.random.randint(1, 21)
        img_font = Image.open("yellow/{0}.png".format(name))
        ran_w = np.random.randint(50, 180)
        img_new = img_font.resize((ran_w, ran_w))

        ran_x1 = np.random.randint(0, 300 - ran_w)
        ran_y1 = np.random.randint(0, 300 - ran_w)

        r, g, b, a = img_new.split()
        background_resize.paste(img_new, (ran_x1, ran_y1), mask=a)

        ran_x2 = ran_x1 + ran_w
        ran_y2 = ran_y1 + ran_w

        background_resize.save(
            "../../source/target_detection/test_pic_plus/{0}{1}.png".format(x, "." + str(ran_x1) + "." + str(ran_y1) +
                                                                            "." + str(ran_x2) + "." + str(
                ran_y2) + ".1"))

        # background_resize.save(
        #     "../../source/target_detection/test_pic_plus/{0}{1}.png".format(x, "." + str(0) + "." + str(0) +
        #                                                                     "." + str(0) + "." + str(
        #         0) + ".0"))

        if x >= 1000:
            print(x)
            break
        x += 1
