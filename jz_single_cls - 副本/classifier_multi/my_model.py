import os
import sys
import copy
import torch
import pandas as pd
from torch import nn
from main_utils import Model
from classifier_multi.training_code import CSVDataset, Train
from classifier_multi.classifier_multi_utils import predict, select_hard_example


class MyModel(Model):

    @staticmethod
    def load_model(model_dir):
        print("===== Loading model...")
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
        m = MyModel(class_name=checkpoint['class_name'], net=checkpoint['net'])

        return m

    def __init__(self, class_name, net, name="efficientnet_b7"):
        self.name = name
        self.class_name = class_name

        if self.name == "efficientnet_b7":
            self.net = net
        else:
            raise Exception("不支持这种模型：{}".format(self.name))

    def save_model(self, save_dir):
        print("Saving model to {}".format(save_dir))
        state = {'net': self.net, 'class_name': self.class_name}
        torch.save(state, os.path.join(save_dir, 'checkpoint.pt'))

    def adjust_model(self, class_num):
        print("===== Adjusting model...")
        if self.name == 'efficientnet_b7':
            input_feature = 2560
            self.net.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(input_feature, class_num))
        else:
            raise Exception("不支持这种模型调整：{}".format(self.name))

    def train_model(self, df_train, df_val, work_dir, is_first_train, **options):
        # read params
        net_config = options

        class_name = net_config["class_name"]
        class_num = net_config["class_num"]
        width = net_config["input_shape"]
        height = net_config["input_shape"]
        input_shape = (height, width)
        optimizer_type = net_config["optimizer"]
        batch_size = net_config["fit_batch_size"]
        epochs = net_config["FIT_epochs"]
        image_col = net_config["image_col"]
        label_col = net_config["label_col"]
        id_col = net_config["id_col"]
        partition_dir = net_config['partition_dir']
        df_unlabeled = net_config["df_unlabeled"]

        # 支持新增数据后重新进行智能标注的操作，不会丢掉权重信息
        if is_first_train and self.class_name != class_name:
            # load weights and freeze
            self.adjust_model(class_num)
            self.class_name = class_name
            if self.name == 'efficientnet_b7':
                for param in self.net.named_parameters():  # 冻结参数
                    if not param[0].startswith('classifier'):
                        param[1].requires_grad_(False)
            else:
                raise Exception("暂不支持模型--{}".format(self.name))
        else:
            if self.class_name != class_name:
                sys.exit(
                    "Error:\nmodel class name:\n{}\nis diff data class name:\n{}".format(self.class_name, class_name))

        print("===== Training...")
        if df_val is not None:
            df_val = CSVDataset(df_val, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                                is_train=False)
        df_train = CSVDataset(df_train, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                              is_train=True)
        df_unlabeled = CSVDataset(df_unlabeled, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                                  is_train=True)
        train = Train(df_train, df_val,df_unlabeled, self.net, batch_size, optimizer_type, epochs)
        self.net = train()

    def predict(self, df_img, work_dir, **options):
        print("===== Predicting...")

        df_unlabeled = copy.deepcopy(df_img)
        return predict(df_unlabeled, self.net, options)[0]

    def query_hard_example(self, df_img, work_dir, query_cnt=100, strategy="LeastConfidence", **options):
        print("===== Selecting...")

        df_unlabeled = copy.deepcopy(df_img)
        df_pred, probs = predict(df_unlabeled, self.net, options)
        # select
        df_result = select_hard_example(df_pred, probs, strategy, query_cnt)
        return df_result

    def split_train_val(self, df_init, df_anno, is_first_train, val_size=0.2, random_seed=1):
        if val_size > 0:
            from sklearn.model_selection import train_test_split
            df_train, df_val = train_test_split(df_init, test_size=val_size, random_state=random_seed, stratify=df_init["label"])
            if df_anno is not None:
                df_anno_changed = pd.DataFrame()
                df_val_changed = pd.DataFrame()
                for i in range(0, len(df_anno), 10):
                    df_anno_part, df_val_add = train_test_split(df_anno[i:i + 10], test_size=val_size,
                                                                random_state=random_seed)
                    df_anno_changed = pd.concat([df_anno_part, df_anno_changed], axis=0, ignore_index=True)
                    df_val_changed = pd.concat([df_val_add, df_val_changed], axis=0, ignore_index=True)
                df_anno = df_anno_changed
                df_val = pd.concat([df_val_changed, df_val], axis=0, ignore_index=True)
            # 输出训练集和测试集的标签比例
            print(df_val)
            print("Test label ratio:\n", df_val["label"].value_counts(normalize=True))
            # exit()
        else:
            df_train = df_init
            df_val = None
        if not is_first_train:
            assert df_anno is not None
            df_train = pd.concat([df_train, df_anno], axis=0, ignore_index=True)
        print("Train count: {}\n"
              "Val   count: {}".format(df_train.shape[0], 0 if df_val is None else df_val.shape[0]))
        return df_train, df_val
