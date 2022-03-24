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
import random
import sys
from collections import defaultdict

from jaccard_distance_calculator import JaccardDistanceCalculator
from edit_distace_calculator import EditDistanceCalculator


def load_sentence(path):
    sentences = list(map(str.strip, open(path, encoding="utf8")))
    print("获取句子数量：", len(sentences))
    return sentences


def main():
    sentences = load_sentence("titles.txt")  # 加载所有标题
    sentences = list(sentences)
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    sentence_num = len(sentences)
    avg_sentence_num_per_cluster = int(sentence_num / n_clusters)
    print("指定聚类数量：", n_clusters, ",平均每类句子数量：", avg_sentence_num_per_cluster)
    # distance_calculator = JaccardDistanceCalculator(sentences)
    distance_calculator = EditDistanceCalculator(sentences)

    centers = set()
    while len(centers) < n_clusters:
        centers.add(random.choice(range(n_clusters)))

    max_iter_times = 20
    iter_time = 0
    cluster_res = None
    while iter_time < max_iter_times:
        cluster_res = defaultdict(list)  # 每次迭代前清空迭代结果
        print(f"迭代{iter_time}.........")
        for i in range(sentence_num):
            min_distance = sys.maxsize
            belonged_center = None
            for center in centers:
                if i == center:
                    continue
                distance = distance_calculator[(center, i)]
                if distance < min_distance:
                    min_distance = distance
                    belonged_center = center

            cluster_res[belonged_center].append(i)
        iter_time += 1

        new_centers = distance_calculator.assign_new_centers(cluster_res)
        if new_centers == centers:
            print("聚类中心不再变化，迭代完成！")
            break

        # print("old_centres:", sorted(list(centers)))
        # print("new_centres:", sorted(list(new_centers)))
        centers = new_centers

    cluster_avg_distance = distance_calculator.compute_cluster_avg_distance(cluster_res)
    sorted_cluster_avg_distances = sorted(
        cluster_avg_distance.items(), key=lambda item: item[1]
    )
    print("-"*100)
    for index, label_avg_dis in enumerate(sorted_cluster_avg_distances):
        label, avg_dis = label_avg_dis
        print(f"TOP{index+1}, cluster:{label}, distance:{avg_dis}")
        cluster_indexes = cluster_res.get(label)
        for i in cluster_indexes[:10]:  # 随便打印几个，太多了看不过来
            sentence = sentences[i]
            print(sentence.replace(" ", ""))
        print("-" * 100)


if __name__ == "__main__":
    main()
