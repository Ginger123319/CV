import sys
import copy
from torch import nn
from main_utils import Model
from classifier_multi.training_code import CSVDataset, Train
from main_utils import predict


class MyModel(Model):

    def __init__(self, class_name, net, name="efficientnet_b7"):
        self.name = name
        self.class_name = class_name

        if self.name == "efficientnet_b7":
            self.net = net
        else:
            raise Exception("不支持这种模型：{}".format(self.name))

    def adjust_model(self, class_num):
        print("===== Adjusting model...")
        if self.name == 'efficientnet_b7':
            input_feature = 2560
            self.net.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(input_feature, class_num))
        else:
            raise Exception("不支持这种模型调整：{}".format(self.name))

    def train_model(self, df_train, df_val, is_first_train, **options):
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
        weights_path = net_config["weights_path"]

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
            df_val = CSVDataset(df_val, class_name, input_shape, image_col, label_col, id_col,
                                is_train=False)
        df_train = CSVDataset(df_train, class_name, input_shape, image_col, label_col, id_col,
                              is_train=True)
        train = Train(df_train, df_val, self.net, batch_size, optimizer_type, epochs)
        self.net = train(weights_path)

    def predict(self, df_img, **options):
        print("===== Predicting...")

        df_unlabeled = copy.deepcopy(df_img)
        return predict(df_unlabeled, self.net, options)[0]
