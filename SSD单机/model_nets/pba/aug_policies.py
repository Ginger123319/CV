__all__ = ['get_sub_policies', "apply_augment"]

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
import random

random_mirror = True
VERBOSE = False


def __operation_filter_coordinates(coordinates, labels, w, h):
    boxes_new = []
    labels_new = []
    for i in range(0, len(coordinates), 4):
        box_index = i // 4
        vertex = coordinates[i:i + 4]
        x_s = [v[0] for v in vertex]
        y_s = [v[1] for v in vertex]
        x_min, x_max, y_min, y_max = min(x_s), max(x_s), min(y_s), max(y_s)
        if x_min >= 0 and x_max < w and y_min >= 0 and y_max < h:
            boxes_new.append([x_min, y_min, x_max, y_max])
            labels_new.append(labels[box_index])
        else:
            x_min = max(x_min, 0)
            x_max = min(x_max, w-1)
            y_min = max(y_min, 0)
            y_max = min(y_max, h-1)
            if x_max-x_min > 0 and y_max-y_min>0:
                boxes_new.append([x_min, y_min, x_max, y_max])
                labels_new.append(labels[box_index])

    return boxes_new, labels_new


def __operation_affine(affine_reverse, sample):
    if not isinstance(sample, dict):
        return sample.transform(sample.size, PIL.Image.AFFINE, affine_reverse)

    image = sample["image"]
    target = sample["target"]
    sample["image"] = image.transform(image.size, PIL.Image.AFFINE, affine_reverse)
    if isinstance(target, dict):
        if "boxes" in target and target["boxes"] is not None and len(target["boxes"]) > 0:
            assert len(target["labels"]) == len(target["boxes"])

            boxes = target["boxes"]
            labels = target["labels"]
            w, h = image.size

            a, b, c, d, e, f = affine_reverse
            affine_real = np.array([[e, -b, f * b - c * e], [-d, a, c * d - f * a]]) / (a * e - b * d)
            arr_tmp = np.array([box for box in boxes])
            # areas_ratio = abs(arr_tmp[:, 2] - arr_tmp[:, 0]) * (arr_tmp[:, 3] - arr_tmp[:, 1])/(w*h)
            coordinates_origin = np.concatenate([arr_tmp, arr_tmp[:, [0]], arr_tmp[:, [3]], arr_tmp[:, [2]], arr_tmp[:, [1]]], axis=1).reshape([-1, 2])
            arr = np.concatenate([coordinates_origin, np.ones((coordinates_origin.shape[0], 1))], axis=1)
            arr_new = np.matmul(affine_real, arr.T).T
            coordinates_new = arr_new.reshape(-1, 2).astype(int).tolist()

            # affine_a = np.array([affine_reverse[:3], affine_reverse[3:], [0, 0, 1]])
            # # print("affine reverse:\n", affine_a)
            # affine_real = np.linalg.inv(affine_a)
            # # print("affine real:\n", affine_real)
            # arr_tmp = np.array([box for box in boxes])
            # coordinates_origin = np.concatenate([arr_tmp, arr_tmp[:, [0]], arr_tmp[:, [3]], arr_tmp[:, [1]], arr_tmp[:, [2]]], axis=1).reshape([-1, 2])
            # arr = np.concatenate([coordinates_origin, np.ones((coordinates_origin.shape[0], 1))], axis=1)
            # # print("coordinates origin:\n", coordinates_origin)
            # arr_new = np.matmul(affine_real[:2], arr.T).T
            # # print("coordinates new:\n", arr_new)
            # coordinates_new = arr_new.reshape(-1, 2).astype(int).tolist()

            boxes_new, labels_new = __operation_filter_coordinates(coordinates_new, labels, w, h)
            target["boxes"] = boxes_new
            target["labels"] = labels_new

    return sample


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return __operation_affine(affine_reverse=(1, v, 0, 0, 1, 0), sample=img)


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return __operation_affine(affine_reverse=(1, 0, 0, v, 1, 0), sample=img)


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0] if not isinstance(img, dict) else v * img["image"].size[0]
    return __operation_affine(affine_reverse=(1, 0, v, 0, 1, 0), sample=img)


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1] if not isinstance(img, dict) else v * img["image"].size[1]
    return __operation_affine(affine_reverse=(1, 0, 0, 0, 1, v), sample=img)


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return __operation_affine(affine_reverse=(1, 0, v, 0, 1, 0), sample=img)


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return __operation_affine(affine_reverse=(1, 0, 0, 0, 1, v), sample=img)


def __operation_rotate(angle_degree, sample):
    if not isinstance(sample, dict):
        return sample.rotate(angle_degree)

    image = sample["image"]
    target = sample["target"]

    sample["image"] = image.rotate(angle_degree)

    if isinstance(target, dict):
        if "boxes" in target and target["boxes"] is not None and len(target["boxes"]) > 0:
            assert len(target["labels"]) == len(target["boxes"])
            boxes = target["boxes"]
            labels = target["labels"]
            w, h = image.size
            center_x, center_y = w / 2, h / 2
            import math
            EPS = 0.01
            coordinates = []
            for x_min, y_min, x_max, y_max in boxes:
                coordinates.extend([(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)])
            coordinates_new = []
            for x, y in coordinates:
                d_w, d_h = x - center_x, y - center_y
                r = math.sqrt(d_w ** 2 + d_h ** 2)
                if r < EPS:
                    coordinates_new.append((x, y))
                    continue
                theta = math.pi / 2 * d_h / abs(d_h) if abs(d_w) < EPS else math.atan(d_h / d_w)
                if d_w < 0:
                    theta += math.pi
                theta_new = theta - angle_degree / 180 * math.pi
                x_new = center_x + r * math.cos(theta_new)
                y_new = center_y + r * math.sin(theta_new)
                coordinates_new.append((round(x_new), round(y_new)))

            boxes_new, labels_new = __operation_filter_coordinates(coordinates_new, labels, w, h)
            target["boxes"] = boxes_new
            target["labels"] = labels_new
    return sample


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return __operation_rotate(v, img)


def AutoContrast(img, _):
    if not isinstance(img, dict):
        return PIL.ImageOps.equalize(img)
    img["image"] = PIL.ImageOps.equalize(img["image"])
    return img


def Invert(img, _):
    if not isinstance(img, dict):
        return PIL.ImageOps.invert(img)
    img["image"] = PIL.ImageOps.invert(img["image"])
    return img


def Equalize(img, _):
    if not isinstance(img, dict):
        return PIL.ImageOps.equalize(img)
    img["image"] = PIL.ImageOps.equalize(img["image"])
    return img


def __operation_mirror(sample):
    if not isinstance(sample, dict):
        return PIL.ImageOps.mirror(sample)
    sample["image"] = PIL.ImageOps.mirror(sample["image"])
    target = sample["target"]
    w, h = sample["image"].size
    if isinstance(target, dict):
        if "boxes" in target and target["boxes"] is not None and len(target["boxes"]) > 0:
            assert len(target["labels"]) == len(target["boxes"])
            boxes_new = []
            for x1, y1, x2, y2 in target["boxes"]:
                boxes_new.append([w - x1, y1, w - x2, y2])
            target["boxes"] = boxes_new
    return sample


def Flip(img, _):  # not from the paper
    if not isinstance(img, dict):
        return PIL.ImageOps.mirror(img)
    return __operation_mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    if not isinstance(img, dict):
        return PIL.ImageOps.solarize(img, v)
    img["image"] = PIL.ImageOps.solarize(img["image"], v)
    return img


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    if not isinstance(img, dict):
        return PIL.ImageOps.posterize(img, v)
    img["image"] = PIL.ImageOps.posterize(img["image"], v)
    return img


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    if not isinstance(img, dict):
        return PIL.ImageOps.posterize(img, v)
    img["image"] = PIL.ImageOps.posterize(img["image"], v)
    return img


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if not isinstance(img, dict):
        return PIL.ImageEnhance.Contrast(img).enhance(v)
    img["image"] = PIL.ImageEnhance.Contrast(img["image"]).enhance(v)
    return img


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if not isinstance(img, dict):
        return PIL.ImageEnhance.Color(img).enhance(v)
    img["image"] = PIL.ImageEnhance.Color(img["image"]).enhance(v)
    return img


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if not isinstance(img, dict):
        return PIL.ImageEnhance.Brightness(img).enhance(v)
    img["image"] = PIL.ImageEnhance.Brightness(img["image"]).enhance(v)
    return img


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if not isinstance(img, dict):
        return PIL.ImageEnhance.Sharpness(img).enhance(v)
    img["image"] = PIL.ImageEnhance.Sharpness(img["image"]).enhance(v)
    return img


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert -0.2 <= v <= 0.2

    def cut_image(img):
        w, h = img.size
        nonlocal v
        v = int(abs(v*w))
        if v<3:
            return img
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img
    if not isinstance(img, dict):
        return cut_image(img)
    img["image"] = cut_image(img["image"])
    return img

def _get_augment_dict(for_autoaug=True):
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, -0.2, 0.2),
    ]
    if for_autoaug:
        l += [
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]

    augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in l}
    return augment_dict


AUGMENT_DICT = _get_augment_dict(True)

OPS_NAMES = [
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'Rotate',
    'AutoContrast',
    'Invert',
    'Equalize',
    'Solarize',
    'Posterize',
    'Contrast',
    'Color',
    'Brightness',
    'Sharpness',
    "Cutout"
]


def _check_ops_config(verbose=False):
    for i, ops in enumerate(OPS_NAMES):
        assert ops in AUGMENT_DICT
        if verbose:
            print("OPS", i + 1, AUGMENT_DICT[ops])


def get_sub_policies(total_depth=2, verbose=False):
    _check_ops_config(verbose)

    sub_policies = []

    def dfs(index=0, sub_policy=[], depth=0):
        if depth == total_depth:
            sub_policies.extend([tuple(sub_policy)])
            return
        for i, ops_name in enumerate(OPS_NAMES):
            if i < index:
                continue
            dfs(i + 1, sub_policy + [ops_name], depth + 1)

    dfs(index=0, sub_policy=[], depth=0)
    if verbose:
        for i, s in enumerate(sub_policies):
            print("SUB-POLICY", i + 1, s)
    return sub_policies


def apply_augment(img, name, level):
    #print("APPLY:", name, level)
    augment_fn, low, high = AUGMENT_DICT[name]
    img_new = augment_fn(img.copy(), level * (high - low) + low)
    return img_new


def generate_policies(middle_magnitude=True):
    def clip(value, floor, ceil):
        return max(min(value, ceil), floor)

    policies = list()
    for _i in range(1):
        for name in OPS_NAMES:
            mag = 0.5 if middle_magnitude else random.random()
            #policies.append([name, clip(abs(random.gauss(0, 0.1)), 0, 1), mag])
            policies.append([name, 0, mag])
    return policies


if __name__ == "__main__":
    get_sub_policies(2, verbose=True)
