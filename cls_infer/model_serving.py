import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
import torchvision.transforms as transforms

from net import model
from class_names import classes


def df_insert_rows(df_origin, df_insert, point):
    return pd.concat([df_origin.iloc[:point, :], df_insert, df_origin.iloc[point:, :]], axis=0).reset_index(drop=True)


def read_rgb_img(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        if img.width < 1 or img.height < 1:
            print(
                "This image has strange height [{}] or width [{}]. Skip it: {}".format(img.height, img.width,
                                                                                       img_path))
            return "Image error: size {}x{}".format(img.height, img.width)
        return img
    except Exception as e:
        print("Error [{}] while reading image. Skip it: {}".format(e, img_path))
        return "Image error: {}".format(str(e))


class ModelServing(object):
    def __init__(self):
        print("自定义模型：Init 模型")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model.load_state_dict(torch.load(os.path.join(base_dir, 'resnet18-f37072fd.pth')))
        self.model = model
        print("自定义模型：模型加载成功。")

        self.classes_ = classes

    def predict(self, X):
        # 图片预处理
        preprocess = transforms.Compose([
            transforms.Resize(256),  # 调整图像大小为256x256
            transforms.CenterCrop(224),  # 中心裁剪为224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

        # 读取并整理图片
        old_index = X.index
        X.reset_index(drop=True, inplace=True)

        imgs = []
        skip_index = []
        for i, img_path in enumerate(X['image_path']):
            img = read_rgb_img(img_path)
            if isinstance(img, str):
                skip_index.append((i, img))
                continue
            imgs.append(preprocess(img))
        images_input = torch.stack(imgs, dim=0)

        # 开始预测
        self.model.eval()
        pred = self.model(images_input)
        pred = torch.softmax(pred, dim=-1)
        pred = pred.detach().cpu().numpy()

        prediction = pd.DataFrame({'prediction': np.argmax(pred, axis=1), "prob": np.max(pred, axis=1)})
        prediction['prediction'] = prediction['prediction'].apply(lambda index: self.classes_[index])

        prediction = pd.concat([prediction, pd.DataFrame(pred, columns=["prob_{}".format(c) for c in self.classes_])],
                               axis=1)

        for i, msg in skip_index:
            prediction = df_insert_rows(prediction, pd.DataFrame({'prediction': [msg]}), i)
        prediction.index = old_index

        return prediction


if __name__ == '__main__':
    inputs = pd.DataFrame({'image_path': [r'D:\python\source\三分类\mosque\1.jpg'], 'label': ['tabby cat']})
    modelSer = ModelServing()
    print(modelSer.predict(inputs))
