import numpy as np
import random
from model_nets.pba.aug_policies import OPS_NAMES, apply_augment
from PIL import Image


def clip(value, floor, ceil):
    return max(min(value, ceil), floor)


def aug_by_config(image, box, aug_config="Random"):
    if image.mode=="RGB":
        # 这里box是ndarray,?x5，image是PIL读取的Image
        cur_boxes = box[:,:4].tolist()
        cur_labels = box[:,4].tolist()
        sample = {"image":image, "target":{"boxes":cur_boxes, "labels":cur_labels}}
        if isinstance(aug_config, list):
            # Do aug here
            cnt = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
            for _ in range(cnt):
                # aug_config is list of [AugName Prob Magnitude]
                aug_name, prob, magnitude = aug_config[random.randint(0, len(aug_config) - 1)]
                if random.random() < prob:
                    sample = apply_augment(sample, aug_name, magnitude)
                    # print("Apply augment: {}, {}".format(aug_name, magnitude))
        elif aug_config=="Random":
            cnt = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
            for _ in range(cnt):
                # aug_config is list of [AugName Prob Magnitude]
                aug_name = OPS_NAMES[random.randint(0, len(aug_config) - 1)]
                magnitude = clip(random.gauss(0.5, 0.2), 0, 1)
                sample = apply_augment(sample, aug_name, magnitude)
        elif isinstance(aug_config, dict):
            # Do randaug here
            for _ in range(aug_config["N"]):
                aug_name = OPS_NAMES[random.randint(0, len(OPS_NAMES) - 1)]
                magnitude = aug_config["M"]
                sample = apply_augment(sample, aug_name, magnitude)
        image = sample["image"]
        if len(sample["target"]["boxes"])>0:
            box = np.concatenate([np.array(sample["target"]["boxes"]), np.array(sample["target"]["labels"]).reshape(-1,1)], axis=1)
        else:
            box = np.zeros((0,5))
    return image, box



def get_random_data(image, box, input_shape, aug_config=None, index=0):
        """实时数据增强的随机预处理"""
        
        iw, ih = image.size
        h, w = input_shape

        if aug_config is not None:
            # from PIL import ImageDraw
            # img_draw = image.copy()
            # d = ImageDraw.Draw(img_draw)
            # colors = ["red", "green", "blue", "pink", "orange"]  # list(ImageColor.colormap.keys())
            # for i, _box in enumerate(box):
            #     color = colors[i % len(colors)]
            #     d.rectangle(_box[:-1].tolist(), outline=color, width=3)
            #     d.text(_box[:2].tolist(), str([-1]), fill=color)
            # from pathlib import Path
            # draw_dir = Path("/home/aps/zk/pba-ssd/output/draws")
            # if not draw_dir.exists():
            #     draw_dir.mkdir(parents=True)
            # img_draw.save(Path(draw_dir, "{}_old.png".format(index)))


            image, box = aug_by_config(image, box, aug_config=aug_config)

            # from PIL import ImageDraw
            # img_draw = image.copy()
            # d = ImageDraw.Draw(img_draw)
            # colors = ["red", "green", "blue", "pink", "orange"]  # list(ImageColor.colormap.keys())
            # for i, _box in enumerate(box):
            #     color = colors[i % len(colors)]
            #     d.rectangle(_box[:-1].tolist(), outline=color, width=3)
            #     d.text(_box[:2].tolist(), str([-1]), fill=color)
            # from pathlib import Path
            # draw_dir = Path("/home/aps/zk/pba-ssd/output/draws")
            # if not draw_dir.exists():
            #     draw_dir.mkdir(parents=True)
            # img_draw.save(Path(draw_dir, "{}_new.png".format(index)))


        iw, ih = image.size
        h, w = input_shape
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data