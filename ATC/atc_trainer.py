from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as f1
from sklearn.metrics import top_k_accuracy_score

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.model import *
from utils.loss import *
from utils.dataset import *

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# hyperparameters
train_pth = 'datasets/atc_data.csv'  # 含有source和target的csv文件
label_pth = 'datasets/atc_label.csv' # 含有固定位置所有label的文件

learning_rate = 1e-4    # 学习率
fl_gamma = 2.5          # focal loss中gamma，gamma越大困难样本贡献的loss越大
hidden_size = 1024      # 隐藏层大小
nhead = 4               # 注意力头数
num_layers = 6          # transformer encoder的层数
af_margin = 0.3         # arcface的边缘，边缘越大不同种类间隔越大
warmup_ratio = 0        # 遇热步数 / 训练总步数
source_max_length = 64  # 输入的默认最大长度
batch_size = 128        # 批量大小

n_epochs = 30           # 总训练的epoch数量
curr_epoch = 0          # 当前epoch，默认为0
checkpoint_path = 'checkpoints/atc_model_20.pt' # 继续训练的路径

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

from os.path import exists
checkpoint_path = checkpoint_path if exists(checkpoint_path) else None
print('resume from', checkpoint_path)

atc_df = pd.read_csv(train_pth)
atc_df.source = atc_df.source.str.strip().str.upper()
# atc_df.drop_duplicates(inplace=True) # 根据情况决定是否去重

# set the minimum of a label's count to be 3
arr = np.array(Counter(atc_df.target).most_common())

idx1 = np.where(arr[:,1].astype('int') == 1)[0]
tgt1 = arr[idx1][:,0]

idx2 = np.where(arr[:,1].astype('int') == 2)[0]
tgt2 = arr[idx2][:,0]

add_df1 = atc_df.query('target in @tgt1')
add_df2 = atc_df.query('target in @tgt2')
atc_df = pd.concat((atc_df, add_df1, add_df1, add_df2))

# label encoding
atc_label = pd.read_csv(label_pth)
le = LabelEncoder()
le.fit(atc_label.label)
label_to_id = dict(zip(le.classes_, le.transform(le.classes_)))
id_to_label = dict(zip(le.transform(le.classes_), le.classes_))

atc_df.target = le.transform(atc_df.target)
num_labels = len(set(atc_df.target))

# save sorted label
atc_label = pd.DataFrame({'label':label_to_id.keys(), 'id':label_to_id.values()})
atc_label.to_csv(label_pth, index=None)

tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
# tokenizer.add_tokens(['A', 'B1', 'B2', 'B3', 'B5', 'B6', 'B7', 'B9', 'B12', 'C', 'D', 'E', 'K', 'TNF', 'α'])
# tokenizer.save_pretrained('./tokenizer')

X, y = atc_df.source.tolist(), atc_df.target.tolist()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)

train_dataset = ATC_dataset(X, y, tokenizer, source_max_length, 'train', device)
valid_dataset = ATC_dataset(X_valid, y_valid, tokenizer, source_max_length, 'valid', device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

model = Model(len(tokenizer), num_labels, hidden_size, nhead, num_layers, margin=af_margin).to(device)
criterion = FocalLoss(fl_gamma)
optimizer = Adam(model.parameters(), lr=learning_rate)

if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    curr_epoch = checkpoint['epoch']

optimizer = Adam(model.parameters(), lr=learning_rate)
total_steps = (len(train_dataset) // batch_size) * (n_epochs - curr_epoch) + 1
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps, num_cycles=0.5)

def accuracy(y_true, y_score, k):

    if k == 1:
        return np.mean(y_true == y_score)
    else:
        return top_k_accuracy_score(y_true, y_score, k=k, labels=list(range(num_labels)))

def f1_score(y_true, y_pred):
    return f1(y_true, y_pred, average='weighted')


train_info = []
val_info = []

for epoch in range(curr_epoch, n_epochs):

    model.train()
    train_loss = []
    train_pbar = tqdm(train_loader)
    train_count = len(train_loader)

    for batch in train_pbar:
        train_pbar.set_description_str(f'[ Train | {epoch + 1:03d}/{n_epochs:03d} ]')

        labels = batch['labels']
        logits = model(batch)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss = loss.detach().item()
        train_loss.append(loss)

        train_count -= 1
        if train_count > 0:
            train_pbar.set_postfix_str(f'loss = {loss:.6f} lr = {scheduler.get_last_lr()[0]}')
        else:
            train_avg_loss = sum(train_loss) / len(train_loss)
            train_info.append([epoch, train_avg_loss])
            train_pbar.set_postfix_str(f'average loss = {train_avg_loss:.6f}')

    with torch.no_grad():
        model.eval()
        val_acc_list = []
        val_pbar = tqdm(valid_loader)
        val_count = len(valid_loader)

        for batch in val_pbar:
            val_pbar.set_description_str(f'[ Valid | {epoch + 1:03d}/{n_epochs:03d} ]')
            logits = model(batch)
            preds = logits.argmax(1)
            acc = (preds == batch['labels']).float().mean().item()
            val_acc_list.append(acc)

            val_count -= 1
            if val_count > 0:
                val_pbar.set_postfix_str(f'acc = {acc:.3f}')
            else:
                val_acc = np.mean(val_acc_list)
                val_info.append([epoch, val_acc])
                val_pbar.set_postfix_str(f'average accuracy = {val_acc:.5f}')

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, f'checkpoints/atc_model_{epoch + 1}.pt')

checkpoint = torch.load(f'checkpoints/atc_model_{n_epochs}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
torch.save(model, f'models/atc_model_{n_epochs}.pt')

atc_clean_df = pd.read_csv('datasets/atc_clean.csv')
atc_clean_df.drop_duplicates(inplace=True)
atc_clean_df.target = le.transform(atc_clean_df.target)
X_clean, y_clean = atc_clean_df.source.tolist(), atc_clean_df.target.tolist()

def predict(model, texts):
    texts = [text.strip().upper() for text in texts]
    dataset = ATC_dataset(texts, None, 'predict')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    logits_list = []

    for batch in tqdm(dataloader):
        logits_list.append(model(batch).cpu().detach().numpy())
    return np.vstack(logits_list)

train_logits = predict(model, X_train)
valid_logits = predict(model, X_valid)
clean_logits = predict(model, X_clean)

def print_predict_info(y_true, logits_list, title):
    y_true = np.array(y_true)
    y_pred = logits_list.argmax(1)

    acc1 = np.mean(y_true == y_pred)
    print(f'{title} accuracy      : {acc1:.04f}')

    acc2 = accuracy(y_true, logits_list, 2)
    print(f'{title} top 2 accuracy: {acc2:.04f}')

    acc3 = accuracy(y_true, logits_list, 3)
    print(f'{title} top 3 accuracy: {acc3:.04f}')

    f1 = f1_score(y_true, y_pred)
    print(f'{title} f1            : {f1:.04f}')

print_predict_info(y_train, train_logits, 'Train')
print_predict_info(y_valid, valid_logits, 'Valid')
print_predict_info(y_clean, clean_logits, 'Clean')