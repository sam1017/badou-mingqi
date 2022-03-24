import abc
from collections import defaultdict
from typing import List


class DistanceCalculator:
    def __init__(self, samples: List[str]) -> None:
        super().__init__()
        self.samples = samples
        self.sample_num = len(samples)
        self.distance_matrix: defaultdict(float) = self.compute_mutual_distance()

    @abc.abstractmethod
    def compute_distance(self, i: int, j: int):
        raise NotImplemented

    def compute_mutual_distance(self) -> defaultdict(float):
        """计算所有句子两两之间的距离。

        :return:
        """
        print("计算距离矩阵....")
        sen_num = self.sample_num
        distance = defaultdict(float)
        for i in range(sen_num - 1):
            for j in range(i + 1, sen_num):
                dis = self.compute_distance(i, j)
                distance[(i, j)] = dis
                distance[(j, i)] = dis

        return distance

    def compute_cluster_avg_distance(self, cluster_res) -> defaultdict(float):
        cluster_avg_distance = defaultdict(float)
        for centre, samples in cluster_res.items():
            cluster_sample_num = len(samples)
            dis_sum = sum(self[(i, centre)] for i in range(cluster_sample_num))
            avg_dis = 0 if not samples else dis_sum / cluster_sample_num
            cluster_avg_distance[centre] = avg_dis

        return cluster_avg_distance

    def __getitem__(self, item):
        return self.distance_matrix.get(item, 0)

    def assign_new_centers(self, cluster_res: defaultdict(list)):
        """计算新的聚类中心

        :param cluster_res:当前聚类结果
        :return:
        """
        new_centres = set()
        for samples in cluster_res.values():
            sum_distances = defaultdict(float)
            for sample in samples:
                for i in range(self.sample_num):
                    sum_distances[sample] += self[(sample, i)]
            new_centre = sorted(sum_distances.items(), key=lambda item: item[1])[0][0]
            new_centres.add(new_centre)
        return new_centres
