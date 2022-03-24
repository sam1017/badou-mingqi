#week4作业

#1、修改word2vec_kmeans文件，使得输出类别按照类内平均距离排序

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)          #使用Word2Vec的load方法加载预训练的模型（输入预训练模型路径）
    return model

def load_sentence(path):
    sentences = set()                    #空集合
    with open(path, encoding="utf8") as f:         #以读的方式打开输入语料
        for line in f:                             #循环每一行
            sentence = line.strip()                #删除句子开头和结尾的空字符
            sentences.add(" ".join(jieba.cut(sentence)))        #调用jieba库切分句子得到一个生成器，将序列中的元素以空格连接生成一个新的字符串，然后对句子去重
    print("获取句子数量：", len(sentences))           #句子数量
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()                 #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)     #创建预训练嵌入维度的全0向量
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v")             #加载词向量模型
    sentences = load_sentence("titles.txt")              #加载所有标题
    vectors = sentences_to_vectors(sentences, model)     #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))          #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)                          #定义一个kmeans类
    kmeans.fit(vectors)                                  #进行聚类计算

    vector_label_dict = defaultdict(list)              #创建一个字典{键:[]}
    for vector, label in zip(vectors, kmeans.labels_):  #取出句子和标签
        vector_label_dict[label].append(vector)         #同标签的放到一起
    #输出类别按照类内平均距离排序
    print("文本向量化后的维度:\n",vectors.shape)
    sample_num_label = {}                                #每个簇内的样本数量
    for label in vector_label_dict.keys():
        sample_num_label[label] =len(vector_label_dict[label])
    print("每个簇的标签和簇内的样本数量:\n",sample_num_label)
    print("聚类后簇的数量",len(sample_num_label))
    centroids = kmeans.cluster_centers_                        #聚类质心
    distance = dict.fromkeys(range(n_clusters), 0)
    for i in range(n_clusters):
        text = vector_label_dict[i]                            #标签为i的样本
        center = centroids[i]                                  #标签为i的样本质心
        for sample in text:
            distance[i] += np.sqrt(np.sum((sample - center)**2))
        distance[i] = distance[i]/sample_num_label[i]
    sorted_distance = sorted(distance.items(), key=lambda distance: distance[1])
    print(sorted_distance)
    sentence_label_dict = defaultdict(list)  # 创建一个字典{键:[]}
    for sentence, label in zip(sentences, kmeans.labels_):   #取出句子和标签
        sentence_label_dict[label].append(sentence)          #同标签的放到一起
    #获得排序后的标签值
    sorted_label = []
    for i in range(n_clusters):
        sorted_label.append(sorted_distance[i][0])
    #print(sorted_label)
    for label in sorted_label:
        print("cluster %s :" % label)
        for i in range(min(10, len(sentence_label_dict[label]))):            #打印10个该类别的句子
            print(sentence_label_dict[label][i].replace(" ", ""))            #删除句子中的空格
        print("---------")

if __name__ == "__main__":
    main()