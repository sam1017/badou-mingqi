#!/usr/bin/env python3
# coding: utf-8

"""
#week4作业
#1、修改word2vec_kmeans文件，使得输出类别按照类内平均距离排序
#2.尝试建立不需要向量化的kmeans文本聚类算法
"""


# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def compute_distance(v1, v2):
    """计算两个向量之间的欧式距离

    :param v1:向量 v1
    :param v2:向量 v2
    :return:
    """
    dis = 0
    for i in range(len(v1)):
        dis += pow(v1[i] - v2[i], 2)

    return math.sqrt(dis)


def compute_cluster_avg_distance(vector_label_dict, kmeans: KMeans):
    """计算类内平均距离

    :param vector_label_dict: 聚类后的句子向量
    :param kmeans: Kmeans实例
    :return: 每个聚类的类内平均距离
    """
    cluster_avg_distances = defaultdict(float)
    centers = kmeans.cluster_centers_
    for label in vector_label_dict:
        vectors = vector_label_dict.get(label)
        sentence_num = len(vectors)
        center = centers[label]
        for vector in vectors:
            cluster_avg_distances[label] += compute_distance(vector, center)

        cluster_avg_distances[label] = round(
            cluster_avg_distances[label] / sentence_num if sentence_num > 0 else 0.0, 4
        )
    return cluster_avg_distances


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for index, sentence_label in enumerate(
        zip(sentences, kmeans.labels_)
    ):  # 取出句子和标签, 存储索引是为了找到对应向量。
        sentence, label = sentence_label
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        vector_label_dict[label].append(vectors[index])

    cluster_avg_distance = compute_cluster_avg_distance(
        vector_label_dict, kmeans
    )
    sorted_cluster_avg_distances = sorted(
        cluster_avg_distance.items(), key=lambda item: item[1]
    )

    for index, label_avg_dis in enumerate(sorted_cluster_avg_distances):
        label, avg_dis = label_avg_dis
        print(f"TOP{index+1}, cluster:{label}, distance:{avg_dis}")
        cluster_sentences = sentence_label_dict.get(label)
        for sentence in cluster_sentences[:10]:  # 随便打印几个，太多了看不过来
            print(sentence.replace(" ", ""))
        print("-" * 100)


if __name__ == "__main__":
    main()
