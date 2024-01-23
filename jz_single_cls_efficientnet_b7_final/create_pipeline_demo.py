import os
import ast
import torch
import shutil
import main_utils
import pandas as pd
from pathlib import Path
from torchvision import models
from main_utils import check_data, cal_acc
from classifier_multi.my_model import MyModel

if __name__ == '__main__':

    # ===============================================
    # 手动输入参数
    input_dir = "exp_dir/input"  # 里面需要有文件：input_label.csv，input_added_label.csv，input_unlabeled.csv
    output_dir = "exp_dir/output"  # 件保存路径
    is_first_train = True
    only_predict = False
    # 需要让True和False时候都能跑通
    query_cnt = 100
    val_size = 0.2
    strategy = "LeastConfidence"
    model_id = '2a2ffd0a-7a90-4032-89c6-27cc73003e31'
    label_col = 'label'

    # ===============================================
    if only_predict:
        df_img = pd.read_csv(Path(input_dir, "input_unlabeled.csv"))

        model_dir_upload = str(Path(output_dir, "model_upload").resolve())
        model_dir_check = str(Path(output_dir, "model_check").resolve())
        work_dir = str(Path(output_dir, "work_dir").resolve())
        result_csv_path = str(Path(output_dir, "output_query.csv").resolve())
        predict_csv_path = str(Path(output_dir, "output_predict.csv").resolve())

        # 加载保存的模型
        model = main_utils.load_pipeline_model(model_url=model_dir_upload, work_dir=work_dir)
        print(model.net)
        for name, params in model.net.named_parameters():
            print(name, params.requires_grad)
        print(model.class_name)
        # exit()
        assert isinstance(model, main_utils.Model)
        # 训练预测使用的参数
        Config = {
            "class_num": len(model.class_name),

            "optimizer": 'Adam',
            'fit_batch_size': 32,
            "FIT_epochs": 100,
            "class_name": model.class_name,
            "input_shape": 64,
            "image_col": "path",
            "label_col": label_col,
            "val_ratio": val_size,
            "id_col": 'id',
            "partition_dir": ''
        }
        # predict
        df_result = model.predict(df_img, work_dir=work_dir, **Config)

        # save result csv
        Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(result_csv_path, encoding='utf-8-sig', index=False)
        cal_acc()
    else:
        # 准备变量
        df_init = pd.read_csv(Path(input_dir, "input_label.csv"))
        # 对标签列进行处理
        df_init.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                                     df_init.loc[:, label_col]]
        df_anno = None if is_first_train else pd.read_csv(Path(input_dir, "input_added_label.csv"))
        # 对标签列进行处理
        if df_anno is not None:
            df_anno.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                                         df_anno.loc[:, label_col]]
            df_cls = pd.concat([df_init, df_anno], axis=0, ignore_index=True)
        else:
            df_cls = df_init
        df_img = pd.read_csv(Path(input_dir, "input_unlabeled.csv"))

        model_dir_upload = str(Path(output_dir, "model_upload").resolve())
        model_dir_check = str(Path(output_dir, "model_check").resolve())
        work_dir = str(Path(output_dir, "work_dir").resolve())
        result_csv_path = str(Path(output_dir, "output_query.csv").resolve())
        predict_csv_path = str(Path(output_dir, "output_predict.csv").resolve())

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 检查数据
        check_data(df_init, label_col)

        with open("cls_1000.txt", mode='r', encoding='utf-8') as f:
            class_name = f.read().splitlines()

        # ===============================================
        # 保存待上传的模型

        net = models.efficientnet_b7(pretrained=False)
        checkpoint = torch.load(r'./efficientnet_b7_lukemelas-dcc49843.pth')

        net.load_state_dict(checkpoint)

        model = MyModel(class_name=class_name, net=net)
        main_utils.save_pipeline_model(model=model, model_id=model_id, save_dir=model_dir_upload)

        # ===============================================
        # 检查模型功能是否正常
        # ===============================================
        # 加载保存的模型
        model = main_utils.load_pipeline_model(model_url=model_dir_upload, work_dir=work_dir)
        assert isinstance(model, main_utils.Model)

        # 目前类别数目、类别名称
        class_num = df_cls[label_col].nunique()
        class_name = list(set(df_cls[label_col].tolist()))
        class_name.sort()

        # split train val
        df_train, df_val, train_ratios = model.split_train_val(df_init, df_anno, is_first_train, val_size=val_size,
                                                               random_seed=1)
        # 训练预测使用的参数
        Config = {
            "class_num": class_num,
            "optimizer": 'Adam',
            'fit_batch_size': 32,
            "FIT_epochs": 100,
            "class_name": class_name,
            "input_shape": 64,
            "image_col": "path",
            "label_col": label_col,
            "val_ratio": val_size,
            "id_col": 'id',
            "partition_dir": '',
            "df_unlabeled": df_img,
            "train_ratios": train_ratios
        }

        # train
        model.train_model(df_train=df_train, df_val=df_val, work_dir=work_dir, is_first_train=is_first_train, **Config)

        # query
        df_result = model.query_hard_example(df_img, work_dir=work_dir, query_cnt=query_cnt, strategy=strategy,
                                             **Config)

        # save result csv
        Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(result_csv_path, index=False)
        cal_acc()

        # persist pipeline
        main_utils.save_pipeline_model(model=model, model_id=model_id, save_dir=model_dir_check)

    # ===============================================
    print("\n\n\n当前only_predict={},还需要测试only_predict={}时的情况".format(only_predict, not only_predict))
    print("如果检查结果正常,可以把生成的模型打包上传")
    print("需要打包上传的Pipeline模型文件是: {}".format(model_dir_upload))
    print("\n\nDone!\n\n")
