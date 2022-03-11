#!/usr/bin/env python
# coding: utf-8

"""
@Author     :1242400788@qq.com
@Date       :2022/3/11
@Desc       :
"""

import torch
from torch.utils.data import Dataset


class TorchTransform(object):
    """
    提取数据集中的回答原文和其对应的标签 id
    """

    def __init__(self):
        """
        接受外部输入的参数，从而初始化内部参数
        """

    def __call__(self, X, Y):
        """
        实现具体处理方法

        @param
        X (list<list<int>>): 数据集，已转id。DataSet 每次调用输入一条数据
        Y (list<int>): X 的标签，已转 id。DataSet 每次调用输入一条数据

        @return
        X (list<list<int>>): 数据集，已转id。DataSet 每次调用输入一条数据
        Y (list<int>): X 的标签，已转 id。DataSet 每次调用输入一条数据
        """

        return X, Y


class TorchDataset(Dataset):

    def __init__(self, X, Y, transform):
        """
        初始化参数

        @param
        X (list<list<int>>): 数据集，已转id
        Y (list<int>): X 的标签，已转 id
        transform: 自定义 Transform 方法
        """

        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        """
        重写方法：接收索引，返回样本

        @param
        index: (int) 被选择的样本的 index

        @return
        x (list<list<int>>): index 对应的数据，已转id
        y (list<int>): x 的标签，已转 id
        """

        x, y = self.transform(X=self.X[index], Y=self.Y[index])

        return x, y

    def __len__(self):
        """
        返回数据集大小
        """

        return len(self.X)


def customized_collate_fn(batch_data):
    """
    自定义输出方法

    @param
    batch_data (list<tuple<x, y>>): [(x1, y1), ...]

    @return
    batch_x ：一个批次的数据
    batch_y (torch.LongTensor, size=(batch_size,h)): batch_x 对应的标签
    """

    batch_x = []
    batch_y = []

    for x, y in batch_data:
        batch_x.append(x)
        batch_y.append(y)

    batch_x = torch.LongTensor(batch_x)
    batch_y = torch.LongTensor(batch_y)

    return batch_x, batch_y
