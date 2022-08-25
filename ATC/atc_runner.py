# 通过调节参数threshold(阈值)（0%~100%）来控制识别的标签是否合理，默认设置为60%，即0.6
# 药品输入长度最大为64，超过部分会被截断
# 仅支持识别单个药品名
# 如果错别字为药品名关键字时或者有多个错别字的时候正确率不高，字符顺序调换影响不大
# 当输出很接近正确分类但不正确时，可以调整k（1，2或3）的值来获取更多可能

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer
from utils.model import *

k = 2
threshold = 0.6
# source_max_length = 64

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def predict(texts, model, tokenizer, threshold=0.6, k=1, guess=False):
    model.eval()
    assert k in (1,2,3)
    atc_label = pd.read_csv('datasets/atc_label.csv')
    le = LabelEncoder()
    le.fit(atc_label.label)
    id_to_label = dict(zip(le.transform(le.classes_), le.classes_))
    texts_, res1, res2, res3 = [text.strip().upper() for text in texts], [], [], []
    input_ids = tokenizer(texts_, padding='max_length', truncation=True, max_length=64)['input_ids']
    logits = model({'input_ids':torch.tensor(input_ids).to(device), 'mode':'predict'}).detach().cpu().numpy()
    for logit in logits:
        if np.max(logit) < threshold:
            res1.append('其它')
            res2.append(f'{id_to_label[np.argmax(logit)]}' if guess else '')
            res3.append('')
        else:
            pred1, pred2, pred3 = logit.argsort()[-3:]
            label1, label2, label3 = id_to_label[pred1], id_to_label[pred2], id_to_label[pred3]
            res1.append(f'{label3} ({min(1., logit[pred3])*100:.01f}%)')
            res2.append(f'{label2} ({min(1., logit[pred2])*100:.01f}%)' if logit[pred2] >= threshold else '')
            res3.append(f'{label1} ({min(1., logit[pred1])*100:.01f}%)' if logit[pred1] >= threshold else '')
    if k == 1:
        return pd.DataFrame({'text':texts, 'predicted':res1})
    elif k == 2:
        return pd.DataFrame({'text':texts, 'first':res1, 'second': res2})
    else:
        return pd.DataFrame({'text':texts, 'first':res1, 'second': res2, 'third':res3})

tokenizer = AutoTokenizer.from_pretrained('tokenizer')
model = torch.load('models/atc_model_20.pt').to(device)

texts = open('input.txt', encoding='utf8').read().split('\n')
pred_info = predict(texts, model, tokenizer, threshold=threshold, k=k, guess=False)
pred_info.to_csv('output.csv', index=None, header=None)
# print(pred_info)

atc_data = pd.read_csv('datasets/atc_clean.csv')
sample = atc_data.sample(n=15)
texts = sample.source.tolist()
expected = sample.target.tolist()
pred_info = predict(texts, model, tokenizer, threshold=threshold, k=k)
pred_info['expected'] = expected
pred_info[''] = [l.split()[0] for l in pred_info['first']] == pred_info['expected']
print(pred_info)

