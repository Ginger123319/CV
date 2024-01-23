import os
import torch
from classifier_multi import Model
from torch import nn
from classifier_multi.training_code import CSVDataset, Train
from classifier_multi.prediction_code import Predict
import numpy as np


class MyModel(Model):

    @staticmethod
    def load_model(model_dir):
        print("===== Loading model...")

        checkpoint = torch.load(os.path.join(model_dir, 'model.pth'))
        model = checkpoint['net']
        model.load_state_dict(checkpoint['weights'])
        my_model = MyModel(checkpoint['name'])
        my_model.model = model
        my_model.out_dim = checkpoint['out_dim']

        return my_model

    def __init__(self, name="MyModel"):
        self.name = name
        self.out_dim = None
        self.weight = None
        self.model = None

        # if self.name == "resnet50":
        #     self.model = models.resnet50(pretrained=False)
        #     self.weight = torch.load(r'/opt/aps/code/project/ced9df28-a330-42f6-8396-84363a312bbb/temp/resnet50-0676ba61.pth')
        #     self.model.load_state_dict(self.weight)
        #     for param in self.model.parameters():#冻结参数
        #         param.requires_grad_(False)

        # elif self.name == "efficientnet_b7":
        #     self.model = models.efficientnet_b7(pretrained=False)
        #     self.weight = torch.load(r'/opt/aps/code/project/c470d522-8627-4b45-a588-33ff87f9e07a/temp/efficientnet_b7_lukemelas-dcc49843.pth')
        #     self.model.load_state_dict(self.weight)
        #     for param in self.model.parameters():#冻结参数
        #         param.requires_grad_(False)

        # else:
        #     raise Exception("不支持这种模型：{}".format(self.name))

    def load_weights(self):
        print("===== Loading weights...")
        # 加载适配的权重
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in self.weight.items() if
                           k in list(model_dict.keys()) and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def adjust_model(self, *args, **kwargs):
        print("===== Adjusting model...")
        class_num = kwargs['class_num']
        self.out_dim = class_num
        if self.name == 'resnet50':
            input_feature = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(input_feature, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, class_num))
        elif self.name == 'efficientnet_b7':
            input_feature = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(input_feature, 512),
                nn.ReLU(),
                nn.Linear(512, class_num))
        else:
            raise Exception("不支持这种模型调整：{}".format(self.name))

    def save_model(self, save_dir):
        print("===== Saving model...")

        state = {'net': self.model, 'weights': self.weight, 'name': self.name, 'out_dim': self.out_dim}
        torch.save(state, os.path.join(save_dir, 'model.pth'))

    def train_model(self, df_train, df_val, net_config=None):

        class_name = net_config["class_name"]
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

        print("===== Training model...")

        train_dataset = df_train
        val_dataset = df_val
        train_dataset = CSVDataset(train_dataset, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                                   is_train=True)
        if val_dataset is not None:
            val_dataset = CSVDataset(val_dataset, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                                     is_train=False)

        train = Train(train_dataset, val_dataset, self.model, batch_size, optimizer_type, epochs)
        self.model, self.weight = train()

    def predict(self, df_img, net_config=None):

        class_name = net_config["class_name"]
        width = net_config["input_shape"]
        height = net_config["input_shape"]
        input_shape = (height, width)
        image_col = net_config["image_col"]
        label_col = net_config["label_col"]
        id_col = net_config["id_col"]
        partition_dir = net_config['partition_dir']

        print("===== Predicting...")

        df_unlabel = df_img
        pred_dataset = CSVDataset(df_unlabel, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                                  is_train=False)
        predict = Predict(pred_dataset, self.model)
        preds = predict.predict_probs()
        return preds


if __name__ == '__main__':
    save_dir = './'
    model = MyModel('resnet50')
    print(model.model.load_state_dict(model.weight))
    model.adjust_model(class_num=30)
    model.save_model(save_dir)
