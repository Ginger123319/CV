import os
import ast
import sys
import torch
import shutil
import main_utils
import pandas as pd
from pathlib import Path
from torchvision import models
from main_utils import check_data
from classifier_multi.my_model import MyModel

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ===============================================
    # 手动输入参数
    # 修改此处控制是否只进行预测
    only_predict = True
    # 修改此处可以改变验证集比例
    val_size = 0.2
    # 修改此处修改初始权重位置
    weights = r'D:/Python/pretrained_weights/checkpoints/efficientnet_b7_lukemelas-dcc49843.pth'
    # ==============================================

    input_dir = "exp_dir/input"  # 里面需要有文件：input_label.csv，input_unlabeled.csv
    output_dir = "exp_dir/output"  # 输出文件保存路径
    is_first_train = True
    label_col = 'label'

    # ===============================================
    if only_predict:
        df_img = pd.read_csv(Path(input_dir, "input_unlabeled.csv"))
        result_csv_path = str(Path(output_dir, "output_query.csv").resolve())
        model_path = r'exp_dir/output/model.pkl'
        weights_path = r'exp_dir/output/best_network.pth'
        class_path = r'exp_dir/output/class.pt'

        # 加载保存的模型
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=DEVICE)
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            else:
                sys.exit("no weights!!!")
        else:
            sys.exit("no model!!!")
        if os.path.exists(class_path):
            class_name = torch.load(class_path, map_location=DEVICE)
        else:
            sys.exit("no class names!!!")

        # 预测使用的参数
        Config = {
            "class_name": class_name,
            "input_shape": 224,
            "image_col": "path",
            "label_col": label_col,
            "id_col": 'id'
        }
        model = MyModel(class_name=class_name, net=model)
        assert isinstance(model, main_utils.Model)
        # predict
        df_result = model.predict(df_img, **Config)

        # save result csv
        Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(result_csv_path, encoding='utf-8-sig', index=False)
    else:
        weights_path = str(Path(output_dir).resolve())

        if os.path.exists(weights_path):
            shutil.rmtree(weights_path)
        os.makedirs(weights_path, exist_ok=True)
        # 准备变量
        df_init = pd.read_csv(Path(input_dir, "input_label.csv"))
        # 对标签列进行处理
        df_init.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                                     df_init.loc[:, label_col]]
        df_cls = df_init

        # 检查数据
        check_data(df_init, label_col)

        with open("cls_1000.txt", mode='r', encoding='utf-8') as f:
            class_name = f.read().splitlines()

        # ===============================================
        # 保存待上传的模型

        net = models.efficientnet_b7(pretrained=False)
        checkpoint = torch.load(weights)
        net.load_state_dict(checkpoint)

        model = MyModel(class_name=class_name, net=net)
        assert isinstance(model, main_utils.Model)

        # 目前类别数目、类别名称
        class_num = df_cls[label_col].nunique()
        class_name = list(set(df_cls[label_col].tolist()))
        class_name.sort()

        # 保存类别名称
        class_path = str(Path(output_dir, "class.pt").resolve())
        if os.path.exists(class_path):
            os.remove(class_path)
        Path(class_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(class_name, class_path)

        # 训练预测使用的参数
        # 此处修改一些训练参数
        Config = {
            "class_num": class_num,
            "optimizer": 'Adam',
            'fit_batch_size': 32,
            "FIT_epochs": 3,
            "class_name": class_name,
            "input_shape": 224,
            "image_col": "path",
            "label_col": label_col,
            "val_ratio": val_size,
            "id_col": 'id',
            "weights_path": weights_path
        }

        # split train val
        df_train, df_val = model.split_train_val(df_init, val_size=val_size, random_seed=1)

        # train
        model.train_model(df_train=df_train, df_val=df_val, is_first_train=is_first_train, **Config)
