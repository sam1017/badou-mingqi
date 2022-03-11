#!/usr/bin/env python
# coding: utf-8

"""
@Author     :1242400788@qq.com
@Date       :2022/3/11
@Desc       :
"""

import random
import string


def build_vocab():
    """
    构建 英文字母 → id 字典

    @return
    letter2id (dict)：英文字母 → id 字典
    """

    letter2id = {}
    for index, char in enumerate(string.ascii_lowercase):
        letter2id[char] = index

    letter2id['unk'] = len(letter2id)  # 未知字母的id

    return letter2id


def build_sample(n_samples, letter2id, sample_length):
    """
    生成数据集

    @param
    n_samples (int)：样本数
    letter2id (dict)：英文字母 → id 字典
    sample_length (int): 样本长度

    @return
    samples (list<list<int>>): 数据集，已转id
    labels (list<int>)：samples 对应的标签，已转id
    """

    samples = []
    labels = []

    for _ in range(0, n_samples, 1):
        # 随机从英文字母中选取 sample_length 个字，可能重复
        sample = [random.choice(list(string.ascii_lowercase)) for _ in range(sample_length)]

        '''
        设置标签
        1. 有 a 或 b，label = 0
        2. 没有 a 或 b，但有 c 或 d，label = 1
        3. 没有 a 或 b，也没有 c 或 d，但有 e 或 f，label = 2
        4. 没有 a 或 b，c 或 d，e 或 f，label = 3
        '''
        if set("ab") & set(sample):
            label = 0
        elif set("cd") & set(sample):
            label = 1
        elif set("ef") & set(sample):
            label = 2
        else:
            label = 3

        # 数据集的英文字母转 id
        sample = [letter2id.get(word, letter2id['unk']) for word in sample]

        samples.append(sample)
        labels.append(label)

    return samples, labels
