# -*- coding: utf-8 -*-
# @Time    : 2022/3/24
# @Author  : baiyunsong
# @Email   : baiyunsong@hikvision.com.cn
# @File    : text_kmeans
# @Software: IDEA
import jieba
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
import math
from collections import defaultdict

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(sentence)
    print("获取句子数量：", len(sentences))
    return list(sentences)

def edit_distance(a,b):
    a_len = len(a)
    b_len = len(b)
    _matrix = [[i + j for j in range(b_len + 1)] for i in range(a_len + 1)]
    res = 0
    if min(a_len,b_len) == 0:
        res = max(a_len,b_len)
    else:
        for i in range(1,a_len + 1):
            for j in range(1,b_len + 1):
                d = 0 if a[i-1] == b[j-1] else 1
                _matrix[i][j] = min(_matrix[i-1][j] + 1,
                                    _matrix[i][j-1] + 1,
                                    _matrix[i-1][j-1] + d
                                    )
        res = _matrix[a_len][b_len]
    return res

def randCent(data, k):
    """ 初始话质心
    :param data:
    :param k:
    :return: centroids
    """
    m = len(data)
    idxs = np.random.choice(a=m, size=k, replace=False, p=None)
    centroids = []
    for i in idxs:
        centroids.append(data[i])
    return centroids


def kmeans(data, k):
    """
    :param data(list): 训练数据
    :param k(int): 类别个数
    :return: centroids_desc（质心）
             subCenter(array):样本所属类别
    """
    m = len(data) #样本个数
    subCenter = np.zeros((m,2)) #初始化每种样本对应的类别
    centroids = randCent(data, k)#初始化质心
    #centroids_desc = np.zeros((m,3))
    change = True #判断是否需要重新计算聚类中心
    num = 0
    while change == True:
        change = False #重置
        num+= 1
        if num>5:
            break
        print("当前轮次：",num)
        for i in range(m):
            minDist = np.inf #初始化样本与聚类中心的最短距离，默认无穷大
            minIndex = 0 #所属类别

            for j in range(k):
                #计算i与每个质心的距离
                dist = edit_distance(data[i],centroids[j])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            #判断是否需要改变
            if subCenter[i,0] != minIndex:
                change = True
                subCenter[i] = [minIndex,minDist]
        #重新计算聚类中心
        for j in range(k):
            sum_all = 0
            r = 0 #每个类别样本个数
            avg_dist = 0
            for i in range(m):
                if subCenter[i, 0] == j:
                    sum_all += subCenter[i,1]
                    r += 1
            try:
                avg_dist = round(sum_all / r)
            except:
                print("r is zero")
            idx = 0
            tmp_dist = np.inf
            for i in range(m):
                if subCenter[i, 0] == j:
                    if subCenter[i,1] >= avg_dist and subCenter[i,1] < tmp_dist:
                        tmp_dist = subCenter[i,1]
                        idx = i
            #centroids[j] = [data[idx],j,avg_dist]
            centroids[j] = data[idx]
    return centroids,subCenter

def calc_avg_dist(subCenter,k,m):
    avg_dists = []
    #计算平局距离
    for j in range(k):
        sum_all = 0
        r = 0 #每个类别样本个数
        avg_dist = 0
        for i in range(m):
            if subCenter[i, 0] == j:
                sum_all += subCenter[i,1]
                r += 1
        avg_dist = round(sum_all / r)
        avg_item = [j,avg_dist]
        avg_dists.append(avg_item)
    return avg_dists


if __name__ == '__main__':
    path = r'./titles.txt'
    sentences = load_sentence(path)
    #print(sentences[:10])
    #centroids = randCent(sentences,4)
    m = len(sentences)
    k = int(math.sqrt(m))
    print("类别数量：",k)
    #k=2
    centroids,subCenter = kmeans(sentences, k)
    acg_dists = calc_avg_dist(subCenter,k,m)
    orders = sorted(acg_dists,key=lambda x:x[1])#根据平均距离排序
    print(orders)
    sentence_label_dict = defaultdict(list)
    for sentence,label in zip(sentences,subCenter[:,0]):
        sentence_label_dict[label].append(sentence)
    for label,avg_dist in orders:
        print("cluster %s :" % label)
        sen = sentence_label_dict[label]
        for i in range(min(10, len(sen))):
            print(sen[i])
        print("---------")





