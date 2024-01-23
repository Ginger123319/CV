import copy
import os
import torch
from torch import nn
from main_utils import Model
from torchvision import models
from classifier_multi.training_code import CSVDataset, Train
from classifier_multi.classifier_multi_utils import predict, select_hard_example


class MyModel(Model):

    @staticmethod
    def load_model(model_dir):
        print("===== Loading model...")
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
        m = MyModel(class_name=checkpoint['class_name'])
        m.weights = checkpoint['weights']
        m.net = checkpoint['net']
        return m

    def __init__(self, class_name, name="resnet50"):
        self.name = name
        self.class_name = class_name
        # self.weights = "TODO"
        if self.name == "resnet50":
            self.net = models.resnet50(pretrained=False)
            self.weights = torch.load(
                r'C:\Users\Ginger\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth')
        else:
            raise Exception("不支持这种模型：{}".format(self.name))

    def save_model(self, save_dir):
        print("Saving model to {}".format(save_dir))
        state = {'net': self.net, 'weights': self.weights, 'class_name': self.class_name}
        torch.save(state, os.path.join(save_dir, 'checkpoint.pt'))

    def adjust_model(self, class_num):
        print("===== Adjusting model...")
        if self.name == 'resnet50':
            input_feature = 2048
            self.net.fc = nn.Sequential(
                nn.Linear(input_feature, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, class_num))
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

        # load weights and freeze
        if is_first_train:
            self.net.load_state_dict(self.weights)
            self.adjust_model(class_num)
        else:
            self.adjust_model(class_num)
            self.net.load_state_dict(self.weights)
        if self.name == 'resnet50':
            for param in self.net.named_parameters():  # 冻结参数
                if not param[0].startswith('fc'):
                    param[1].requires_grad_(False)
        else:
            raise Exception("暂不支持模型--{}".format(self.name))

        print("==== Training...")
        df_train = CSVDataset(df_train, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                              is_train=True)
        if df_val is not None:
            df_val = CSVDataset(df_val, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                                is_train=False)

        train = Train(df_train, df_val, self.net, batch_size, optimizer_type, epochs)
        self.net, self.weights = train()

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
