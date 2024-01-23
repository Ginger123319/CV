import torch
import random
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
    partition_dir = net_config['partition_dir']

    df_unlabel = df_img
    pred_dataset = CSVDataset(df_unlabel, class_name, input_shape, image_col, label_col, id_col, partition_dir,
                              is_train=False)
    pred_obj = Predict(pred_dataset, model)
    preds = pred_obj.predict_probs()

    # 预测结果后处理
    df_unlabel.drop(image_col, axis=1, inplace=True)
    # 确定预测的类别
    out = torch.argmax(preds, dim=1).tolist()
    label = [{"annotations": [{"category_id": class_name[index]}]} for index in out]
    df_unlabel[label_col] = label
    # df_unlabel['isHardSample'] = preds

    return df_unlabel, preds


def select_hard_example(df_pred, probs, strategy_name, query_num):
    if query_num > len(probs):
        query_num = len(probs)
    if strategy_name == "EntropySampling":
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)
        hard_sample_list = uncertainties.sort()[1][:query_num].tolist()
    elif strategy_name == "LeastConfidence":
        uncertainties = probs.max(1)[0]
        hard_sample_list = uncertainties.sort()[1][:query_num].tolist()
    elif strategy_name == "MarginSampling":
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        hard_sample_list = uncertainties.sort()[1][:query_num].tolist()
    elif strategy_name == "RandomSampling":
        p = [i for i in range(len(probs))]
        random.shuffle(p)
        hard_sample_list = random.sample(p, query_num)
    else:
        raise Exception("不支持这种格式: {} 的查询策略".format(strategy_name))

    hard_sample = torch.zeros(len(probs), dtype=torch.int8)
    hard_sample[hard_sample_list] = 1
    df_pred['isHardSample'] = hard_sample

    return df_pred


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
