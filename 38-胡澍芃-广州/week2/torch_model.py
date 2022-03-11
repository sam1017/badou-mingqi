#!/usr/bin/env python
# coding: utf-8

"""
@Author     :1242400788@qq.com
@Date       :2022/3/11
@Desc       :
"""

import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, n_labels, sample_length, lstm_hidden_size):
        """
        初始化参数

        @param
        embedding_dim (int): 英文字母向量维度数
        vocab_size (int): 词典单词数
        n_labels (int)：标签种类的数量
        sample_length (int): 样本长度
        lstm_hidden_size (int): LSTM 隐层的维度

        @param

        """
        super(TorchModel, self).__init__()

        # ================================= 模型结构初始化 ================================
        # 词向量矩阵，vocab_size * embedding_dim
        self.letter_lookup_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=1,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.classifier = nn.Linear(in_features=lstm_hidden_size, out_features=n_labels, bias=True)

        # ================================= 权重初始化 ================================
        nn.init.xavier_normal_(tensor=self.classifier.weight.data, gain=1.0)

    def forward(self, x):
        """
        前向传播

        @param
        x (torch.LongTensor, batch_size * sample_length): 一个批次的数据

        @return
        classification_score ((torch.FloatTensor, batch_size * n_labels)): 每个样本的标签分类得分
        """

        x = self.letter_lookup_table(x)    # batch_size * sample_length * embedding_dim
        lstm_outputs, (lstm_h, lstm_c) = self.lstm(x)     # batch_size * sample_length * lstm_hidden_size
        outputs = lstm_outputs[:, -1, :]   # 取最后一个字的输出，batch_size * lstm_hidden_size
        classification_score = self.classifier(outputs)   # batch_size * n_labels

        return classification_score
