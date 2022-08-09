import os
import traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils import iou

anno_src = r"D:\Python\source\FACE\celebA\Anno\list_bbox_celeba.txt"
img_dir = r"D:\Python\source\FACE\celebA\img\img_celeba\img_celeba"

save_path = r"D:\Python\source\FACE\celebA\save_pic"
validate_path = r"D:\Python\source\FACE\celebA\validate"


# ratio_arr表示正\部分\负样本比例，例如[2,5,3]或者【1，3，1】
def gen_sample(face_size, stop_value, ratio_arr=[1, 3, 1]):
    # print("gen size:{} image" .format(face_size) )
    sum_tmp = sum(ratio_arr)
    positive_num = stop_value // sum_tmp * ratio_arr[0]
    negative_num = stop_value // sum_tmp * ratio_arr[1]
    part_num = stop_value // sum_tmp * ratio_arr[2]

    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")
        float_nums = [(0, 0.1), (0.2, 0.4)]
        is_gen_negative = False
        curSeedIndex = 0
        for _ in range(3):
            for i, line in enumerate(open(anno_src)):
                if i < 2:
                    continue
                try:
                    strs = line.split()
                    image_filename = strs[0].strip()
                    # print(image_filename)
                    image_file = os.path.join(img_dir, image_filename)

                    with Image.open(image_file) as img:
                        img_w, img_h = img.size
                        x1 = float(strs[1].strip())
                        y1 = float(strs[2].strip())
                        w = float(strs[3].strip())
                        h = float(strs[4].strip())
                        x2 = float(x1 + w)
                        y2 = float(y1 + h)

                        px1 = 0  # float(strs[5].strip())
                        py1 = 0  # float(strs[6].strip())
                        px2 = 0  # float(strs[7].strip())
                        py2 = 0  # float(strs[8].strip())
                        px3 = 0  # float(strs[9].strip())
                        py3 = 0  # float(strs[10].strip())
                        px4 = 0  # float(strs[11].strip())
                        py4 = 0  # float(strs[12].strip())
                        px5 = 0  # float(strs[13].strip())
                        py5 = 0  # float(strs[14].strip())

                        if x1 < 0 or y1 < 0 or w < 0 or h < 0:
                            continue
                        boxes = [[x1, y1, x2, y2]]
                        side_len = max(w, h)
                        if (is_gen_negative):
                            low = int(-side_len * 0.55)
                            high = int(-side_len * 0.25)
                            _side_len = side_len + np.random.randint(low, high)
                            _boxes = genNegativeBoxes(img_w, img_h, _side_len)

                            for box in _boxes:
                                _iou = iou(box, boxes)[0]
                                # print(face_size,_iou)
                                if _iou < 0.15:
                                    face_crop = img.crop(box)
                                    face_resize = face_crop.resize((face_size, face_size))
                                    negative_anno_file.write(
                                        "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                                    negative_anno_file.flush()
                                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                    negative_count += 1
                                    if (negative_count >= negative_num): return
                                    # print(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            pass
                        else:
                            cx = x1 + w / 2
                            cy = y1 + h / 2
                            np.random.seed()
                            seed = float_nums[curSeedIndex]
                            if (positive_count >= positive_num and curSeedIndex == 0):
                                curSeedIndex = 1
                                continue
                            elif (part_count >= part_num and curSeedIndex == 1):
                                is_gen_negative = True
                                continue
                            ratio1 = np.random.uniform(seed[0], seed[1])
                            # ratio2=np.random.uniform(seed[0],seed[1])
                            flags = [-1, 1]
                            flag1 = flags[np.random.randint(0, len(flags))]
                            flag2 = flags[np.random.randint(0, len(flags))]
                            low = int(-side_len * 0.35)
                            high = int(side_len * 0.35)
                            if (curSeedIndex == 0):
                                # positive
                                low = int(-side_len * 0.1)
                                high = int(side_len * 0.1)
                            if (low >= high): continue
                            _side_len = side_len + np.random.randint(low, high)
                            _cx = cx + int(cx * flag1 * ratio1)
                            _cy = cy + int(cy * flag2 * ratio1)

                            _x1 = _cx - _side_len / 2
                            _y1 = _cy - _side_len / 2
                            _x2 = _x1 + _side_len
                            _y2 = _y1 + _side_len

                            if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                                continue

                            offset_x1 = (x1 - _x1) / _side_len
                            offset_y1 = (y1 - _y1) / _side_len
                            offset_x2 = (x2 - _x2) / _side_len
                            offset_y2 = (y2 - _y2) / _side_len

                            offset_px1 = 0  # (px1 - x1_) / _side_len
                            offset_py1 = 0  # (py1 - y1_) / _side_len
                            offset_px2 = 0  # (px2 - x1_) / _side_len
                            offset_py2 = 0  # (py2 - y1_) / _side_len
                            offset_px3 = 0  # (px3 - x1_) / _side_len
                            offset_py3 = 0  # (py3 - y1_) / _side_len
                            offset_px4 = 0  # (px4 - x1_) / _side_len
                            offset_py4 = 0  # (py4 - y1_) / _side_len
                            offset_px5 = 0  # (px5 - x1_) / _side_len
                            offset_py5 = 0  # (py5 - y1_) / _side_len

                            crop_box = [_x1, _y1, _x2, _y2]
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size))

                            _iou = iou(crop_box, boxes)[0]
                            # print(face_size,_iou)
                            if _iou > 0.65:
                                if positive_count >= positive_num: continue
                                positive_anno_file.write(
                                    "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {"
                                    "15}\n".format(
                                        positive_count, 1, offset_x1, offset_y1,
                                        offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2,
                                        offset_px3,
                                        offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                                positive_anno_file.flush()
                                face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                                positive_count += 1
                                # print(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            elif 0.6 > _iou > 0.4:
                                if part_count >= part_num: continue
                                part_anno_file.write(
                                    "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                        part_count, 2, offset_x1, offset_y1, offset_x2,
                                        offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                        offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                                part_anno_file.flush()
                                face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                                part_count += 1
                                # print(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            pass
                    print(positive_count + part_count + negative_count)
                    if (positive_count >= positive_num
                            and part_count >= part_num
                            and negative_count >= negative_num): return
                except:
                    traceback.print_exc()
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()


def genNegativeBoxes(w, h, _side_len):
    # 直接左上左下右上右下取4个点
    result = []
    _x1 = 0
    _y1 = 0
    _x2 = _x1 + _side_len
    _y2 = _y1 + _side_len
    result.append([_x1, _y1, _x2, _y2])
    _x1 = w - _side_len
    _y1 = h - _side_len
    _x2 = _x1 + _side_len
    _y2 = _y1 + _side_len
    result.append([_x1, _y1, _x2, _y2])
    _x1 = w - _side_len
    _y1 = 0
    _x2 = _x1 + _side_len
    _y2 = _y1 + _side_len
    result.append([_x1, _y1, _x2, _y2])
    _x1 = 0
    _y1 = h - _side_len
    _x2 = _x1 + _side_len
    _y2 = _y1 + _side_len
    result.append([_x1, _y1, _x2, _y2])
    return result
    pass


if __name__ == '__main__':
    gen_num = 100000
    gen_sample(12, gen_num)
    # gen_sample(24,gen_num)
    # gen_sample(48,gen_num)

    save_path = validate_path
    gen_num = 5000
    # gen_sample(12, gen_num)
    # gen_sample(24, gen_num)
    # gen_sample(48, gen_num)
