#!/usr/bin/env python
# coding: utf-8

"""
@Author     :1242400788@qq.com
@Date       :2022/3/11
@Desc       :
"""

import data_generator
import torch
from torch_model import TorchModel
import pandas as pd

# ================================== 生成测试集 ======================================
letter2id = data_generator.build_vocab()
testset, labels = data_generator.build_sample(n_samples=5, letter2id=letter2id, sample_length=5)

# ================================== 加载模型 ======================================
model = TorchModel(embedding_dim=10, vocab_size=len(letter2id), n_labels=4, sample_length=5, lstm_hidden_size=10)
model.load_state_dict(torch.load(f='./model/model_epoch_2.bin'))
model.eval() # Sets the module in evaluation mode
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # GPU
# model = model.to(device)

# ================================== 预测 ======================================
Label2ID = {'a or b': 0, 'c or d': 1, 'e or f': 2, 'others': 3}
ID2Label = {value: key for key, value in Label2ID.items()}
classification_score = model(x=torch.LongTensor(testset))  # 模型预测
prediction_list = torch.argmax(input=classification_score, dim=1).numpy().tolist()

# ================================== 输出结果 ======================================
result = []
for i in range(0, len(testset), 1):
    result.append({'Data': testset[i], 'Prediction': prediction_list[i], 'Label': labels[i]})

print(pd.DataFrame(data=result))
'''
预测结果如下：
                   Data  Prediction  Label
0      [4, 1, 0, 12, 3]           0      0
1    [6, 12, 11, 18, 3]           1      1
2  [17, 25, 14, 18, 21]           3      3

3  [12, 25, 24, 20, 12]           3      3
4    [15, 11, 8, 24, 5]           2      2
'''