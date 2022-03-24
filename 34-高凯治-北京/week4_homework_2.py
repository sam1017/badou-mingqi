#week4作业

#2.尝试建立不需要向量化的kmeans文本聚类算法
import numpy as np
import random
import sys
import re
import Levenshtein
import copy
import operator

def load_sentence(path):
    sentences = set()                    #空集合
    with open(path, encoding="utf8") as f:         #以读的方式打开输入语料
        for line in f:                             #循环每一行
            sentence = line.strip()                #删除句子开头和结尾的空字符
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            sentences.add(re.sub(reg, "", sentence))        #只保留中文、大小写字母和阿拉伯数字，然后对句子去重
    print("获取句子数量：", len(sentences))           #句子数量
    return list(sentences)


class KMeansClusterer:                            #k均值聚类
    def __init__(self, sentences, cluster_num):
        self.sentences = sentences                #聚类语料
        self.cluster_num = cluster_num            #聚类簇数
        self.points = self.__pick_start_point(sentences, cluster_num)        #挑选起始聚类点

    def cluster(self):                        #执行聚类
        result = []
        for i in range(self.cluster_num):
            result.append([])                 #生成二维列表，第一个维度为质心数量
        for item in self.sentences:           #循环语料中的每条文本
            distance_min = sys.maxsize        #2^63-1=9223372036854775807
            index = -1                        #索引=-1
            for i in range(len(self.points)):                     #循环每个聚类质心
                distance = Levenshtein.distance(item, self.points[i])                   #计算语料中每条文本和第i个质心的距离
                if distance < distance_min:                       #如果距离小于之前的最小距离
                    distance_min = distance                       #更新最小距离
                    index = i                                     #索引=i
            result[index] = result[index] + [item]                #结果中的第i个质心对应的簇中添加一条文本
        new_center = []                                           #新聚类中心
        for item in result:                                       #循环每个簇
            new_center.append(self.__center(item))                #计算每个簇内部到簇内其他词编辑距离最小的词
        # 中心点未改变，说明达到稳态，结束递归
        if operator.eq(self.points, new_center):                  #如果所有聚类之心没有改变
            sum = self.__sumdis(result)                           #对所有簇中所有文本到各自质心的距离求和
            return result, self.points, sum                       #返回聚类结果、质心、距离和
        self.points = new_center                                  #更新质心
        return self.cluster()                                     #再次调用聚类函数进行聚类

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):         #循环每个簇
            for j in range(len(result[i])):       #循环该簇中的每条文本
                sum += Levenshtein.distance(result[i][j], self.points[i])        #计算每条文本到质心的距离求和
        return sum

    def __center(self, texts):
        # 计算每个簇内部到簇内其他词编辑距离最小的词
        distance_all = [[0] * (len(texts) - 1) for _ in range(len(texts))]
        for i, x in enumerate(texts):
            texts_copy = copy.deepcopy(texts)
            texts_copy.remove(x)
            for j, y in enumerate(texts_copy):
                distance = Levenshtein.distance(x, y)
                distance_all[i][j] = distance
        index_new_center = np.argmin(np.sum(distance_all, axis=1))
        new_center = texts[index_new_center]
        return new_center

    def __pick_start_point(self, sentences, cluster_num):          #挑选起始聚类点
        if cluster_num < 0 or cluster_num > len(sentences):    #如果聚类簇数<0或者聚类数大于语料中的文本数量，则提示错误
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, len(sentences), step=1).tolist(), cluster_num)
        #多个字符中生成指定数量的随机字符。从所有文本中随机挑选10条文本的索引作为起始聚类点
        points = []
        for index in indexes:
            points.append(sentences[index])        #二维列表，把起始聚类点对应文本转为列表添加到points中
        return points

sentences = load_sentence("titles.txt")              #加载所有标题,以列表形式存储文本字符串
kmeans = KMeansClusterer(sentences, 10)              #初始化聚类函数，聚为10类
result, centers, distances = kmeans.cluster()       #调用聚类中的cluster方法完成聚类
print(result)
for i in range(10):
    print(len(result[i]))
print(centers)
print(distances)