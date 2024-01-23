import abc
import torch
from collections import Counter
from classifier_multi.prediction_code import Predict
from classifier_multi.training_code import CSVDataset


def predict(df_img, model, net_config=None):
    class_name = net_config["class_name"]
    width = net_config["input_shape"]
    height = net_config["input_shape"]
    input_shape = (height, width)
    image_col = net_config["image_col"]
    label_col = net_config["label_col"]
    id_col = net_config["id_col"]

    df_unlabel = df_img
    pred_dataset = CSVDataset(df_unlabel, class_name, input_shape, image_col, label_col, id_col,
                              is_train=False)
    pred_obj = Predict(pred_dataset, model)
    preds = pred_obj.predict_probs()
    # 确定预测的类别
    out = torch.argmax(preds, dim=1).tolist()
    label = [{"annotations": [{"category_id": class_name[index]}]} for index in out]
    df_unlabel[label_col] = label

    return df_unlabel, preds


def show_gpu_info():
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Available cuda:{i}\n", torch.cuda.get_device_properties(f"cuda:{i}"))
    else:
        print("cuda not available.")


def check_data(df_label, label_col):
    c = Counter(df_label[label_col])
    if len(c) < 2:
        raise Exception("分类的类别至少是2，现在为：{}".format(len(c)))
    break_flag = False
    for k, v in c.items():
        # print("类别[{}]个数：{}".format(k, v))
        if v < 2:
            break_flag = True
    if break_flag:
        raise Exception("每个类别的样本数至少为2！")


class Model(abc.ABC):

    @abc.abstractmethod
    def train_model(self, df_train, df_val, is_first_train, **options):
        pass

    @abc.abstractmethod
    def predict(self, df_img, **options):
        pass

    @staticmethod
    def split_train_val(df_init, val_size=0.2, random_seed=1):
        if val_size > 0:
            from sklearn.model_selection import train_test_split
            df_train, df_val = train_test_split(df_init, test_size=val_size, random_state=random_seed)
        else:
            df_train = df_init
            df_val = None
        print("Train count: {}\n"
              "Val   count: {}".format(df_train.shape[0], 0 if df_val is None else df_val.shape[0]))
        return df_train, df_val
