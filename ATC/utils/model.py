import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# 由于我们的模型不包含递归和卷积，为了使模型能够利用序列顺序，我们标记序列中的相对或绝对位置的信息

# PE(pos, 2i) = sin(pos / 10000^(2i / d_model) )
# PE(pos, 2i) = sin(pos / 10000^(2i / d_model) )
# 2i: torch.arange(0, d_model, 2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # PE shape: (max_len, d_model)
        self.PE = torch.zeros(1, max_len, d_model)

        # 创建⼀个⾜够⻓的pos
        pos = torch.arange(0, max_len, 1.).view(-1, 1)

        # div: 10000^(2i / d_model)
        div = torch.pow(10000, (torch.arange(0, d_model, 2.) / d_model))

        self.PE[:, :, 0::2] = torch.sin(pos / div)
        self.PE[:, :, 1::2] = torch.cos(pos / div)

    def forward(self, X):
        # X.shape: [batch_size, seq_lengh, d_model]
        # 残差连接, 每一个batch都要加, PE shape改为(1, max_len, d_model)
        X = X + self.PE[:, :X.size(1), :].to(X.device)
        return self.dropout(X)
    
# https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
class Model(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 num_labels, 
                 hidden_size, 
                 nhead, 
                 num_layers, 
                 dropout=0.1, 
                 margin=0.5
    ):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size) # 稠密表示所有词
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward=hidden_size*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.arcface = ArcFace(hidden_size, num_labels, margin=margin)

    def forward(self, inputs):
        # input_ids -> torch.tensor
        input_ids = inputs['input_ids']
        
        # [PAD]的id为0，则input_ids中等于0的项为src_padding_mask
        src_padding_mask = (input_ids == 0).to(input_ids.device)
        src = self.embedding(input_ids) * np.sqrt(self.hidden_size) # 回调分布至N(0,1)，之前embedding的时候归一化除以了√embedding_dim
        src = self.positional_encoding(src)
        
        # 获取CLS对应的hidden state
        hidden_state = self.transformer(src, None, src_padding_mask)[:,0,:]

        if inputs['mode'] == 'train':
            return self.arcface(hidden_state, inputs['labels'])
        else:            
            hidden_state = F.normalize(hidden_state)
            weights = F.normalize(self.arcface.weights.T)
            return F.relu(torch.matmul(hidden_state, weights)) * 3 # 乘以的3是放大系数
            
            
# https://cloud.tencent.com/developer/article/1811179
class ArcFace(nn.Module):
    
    def __init__(self,in_features,out_features, margin = 0.5 ,scale = 20):
        super(ArcFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        self.weights = nn.Parameter(torch.FloatTensor(out_features,in_features))
        nn.init.xavier_normal_(self.weights)
        
    def forward(self,features,targets):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weights), bias=None) # 计算两个标准向量的夹角
        cos_theta = cos_theta.clip(-1, 1) # cos的范围（-1，1）
        
        arc_cos = torch.acos(cos_theta)
        M = F.one_hot(targets, num_classes = self.out_features) * self.margin
        arc_cos = arc_cos + M
        
        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.scale
        return logits
