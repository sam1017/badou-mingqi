#!/usr/bin/env python
# coding: utf-8

"""
@Author     :1242400788@qq.com
@Date       :2022/3/11
@Desc       :
"""

import data_generator
from torch_model import TorchModel
import torch
from preprocess import TorchTransform, TorchDataset, customized_collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm


# ================================== 生成数据 ======================================
letter2id = data_generator.build_vocab()
dataset, labels = data_generator.build_sample(n_samples=10000, letter2id=letter2id, sample_length=5)
trainset_X = dataset[0:8500]
trainset_Y = labels[0:8500]
valset_X = dataset[8501:]
valset_Y = labels[8501:]

# ================================ 数据预处理 =======================================
transform = TorchTransform()
trainDataSet = TorchDataset(X=trainset_X, Y=trainset_Y, transform=transform)
trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=32, shuffle=True, drop_last=False, collate_fn=customized_collate_fn)
valDataSet = TorchDataset(X=valset_X, Y=valset_Y, transform=transform)
valDataLoader = DataLoader(dataset=valDataSet, batch_size=32, shuffle=False, drop_last=False, collate_fn=customized_collate_fn)

# ================================ 构建模型 =======================================
model = TorchModel(embedding_dim=10, vocab_size=len(letter2id), n_labels=4, sample_length=5, lstm_hidden_size=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ================================ 损失函数 =======================================
loss_function = nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')    # 损失函数：交叉熵

# ================================ 优化器 =======================================
optimizer = optim.Adam(params=model.parameters(), lr=0.001, eps=1e-8, weight_decay=0)

# ================================ 迭代训练 =======================================
Label2ID = {'a or b': 0, 'c or d': 1, 'e or f': 2, 'others': 3}
labels = list(Label2ID.values())
target_names = list(Label2ID.keys())
num_labels = len(Label2ID)
max_epoch = 3

for epoch in range(0, max_epoch, 1):
    epoch_loss = 0.0  # 每个 epoch 的损失
    prediction_lists = []  # 每个 epoch 所有样本的预测结果
    label_lists = []  # 每个 epoch 所有样本的真实值

    # ==========================================  训练集：训练 ===================================
    model.train()  # Sets the module in training mode
    for batch_id, batch_data in enumerate(tqdm(trainDataLoader)):
        batch_x, batch_y = batch_data

        # 数据迁移至 GPU
        # batch_x = batch_x.to(device)
        # batch_y = batch_y.to(device)

        optimizer.zero_grad()  # 梯度清零

        classification_score = model(x=batch_x)  # 模型预测

        # 计算每个批次的损失
        batch_loss = loss_function(classification_score, batch_y.view(-1))
        epoch_loss = epoch_loss + batch_loss.item()  # 累加 batch_loss 计算 epoch_loss

        # 数据迁移至 CPU
        # classification_score = classification_score.to('cpu')
        # batch_y = batch_y.to('cpu')

        # 收集用于计算训练集 classification report 的数据
        prediction_list = torch.argmax(input=classification_score, dim=1).numpy().tolist()
        label_list = batch_y.squeeze().numpy().tolist()
        prediction_lists.extend(prediction_list)
        label_lists.extend(label_list)

        batch_loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新
        torch.cuda.empty_cache()  # 清除 GPU 上不再被引用的变量

    # 保存每次 epoch 的模型
    torch.save(obj=model.state_dict(), f='./model/model_epoch_{}.bin'.format(epoch))

    # ==========================================  训练集：验证 ===================================
    print('训练集 epoch {}:  loss_sum={}'.format(epoch, epoch_loss))
    print(classification_report(y_pred=prediction_lists, y_true=label_lists, labels=labels,
                                target_names=target_names))

    # ==========================================  验证集：验证 ===================================
    epoch_loss_val = 0.0  # 验证集每个 epoch 的损失
    prediction_lists_val = []  # 验证集每个 epoch 所有样本的预测结果
    label_lists_val = []  # 验证集每个 epoch 所有样本的真实值

    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():
        for batch_id_val, batch_data_val in enumerate(tqdm(valDataLoader)):
            batch_x_val, batch_y_val = batch_data_val

            # 数据迁移至 GPU
            # batch_x_val = batch_x_val.to(device)
            # batch_y_val = batch_y_val.to(device)

            classification_score_val = model(x=batch_x_val)  # 模型预测
            # 每个批次的损失
            batch_loss_val = loss_function(classification_score_val, batch_y_val.view(-1))
            epoch_loss_val = epoch_loss_val + batch_loss_val.item()  # 累加 batch_loss 计算 epoch_loss

            # 数据迁移至 CPU
            # classification_score_val = classification_score_val.to('cpu')
            # batch_y_val = batch_y_val.to('cpu')

            torch.cuda.empty_cache()  # 清除 GPU 上不再被引用的变量

            # 收集用于计算验证集 classification report 的数据
            prediction_list_val = torch.argmax(input=classification_score_val, dim=1).numpy().tolist()
            label_list_val = batch_y_val.squeeze().numpy().tolist()
            prediction_lists_val.extend(prediction_list_val)
            label_lists_val.extend(label_list_val)

    print('验证集 epoch {}:  loss_sum={}'.format(epoch, epoch_loss_val))
    print(classification_report(y_pred=prediction_lists_val, y_true=label_lists_val, labels=labels,
                                target_names=target_names))