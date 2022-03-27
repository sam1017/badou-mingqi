# -*- coding: utf-8 -*-
"""============================================================================================================
@Author  : Song Jiazhen
@Data    : 2022/3/25
@Describe: 计算kmeans聚类的类内平均距离，并排序，本程序采用cos距离
==========================================================================================================="""

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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
    vectors = []  # 存放句子向量
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]  # model.wv可按照key-value 形式选取每个词的向量
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化
    sentence_vector_dict = dict(zip(sentences, vectors))  # 存放每个标题对应的向量

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定 义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    # kmeans.labels_中存放每个句向量经过kmeans后所在的聚类标识（类别）
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起

    """计算每个类别下，每个向量到中心向量的距离的平均值"""
    label_dis_dict = defaultdict(int)
    for label in sentence_label_dict.keys():
        label_sentences =  sentence_label_dict[label] # 获取每个类别的所有句子
        distance = 0
        # 获取该类别下中心点向量
        center_vec = kmeans.cluster_centers_[label]
        for sen in label_sentences:
            sen_vec = sentence_vector_dict[sen]
            cos_dis = np.dot(center_vec, sen_vec) / (np.linalg.norm(center_vec)*np.linalg.norm(sen_vec))
            distance += cos_dis
        label_dis_dict[label] = distance / len(label_sentences)

    """对每个中心点的距离进行降序排列，并输出类别"""
    sorted_dis = sorted(label_dis_dict.items(), key=lambda x: x[1], reverse=True)
    for line in sorted_dis:
        print("label %s," %line[0], "distance is %s"  %line[1])


if __name__ == "__main__":
    main()