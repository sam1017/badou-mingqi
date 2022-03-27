from typing import List

from distance_calculator import DistanceCalculator


class JaccardDistanceCalculator(DistanceCalculator):
    def __init__(self, samples: List[str]) -> None:
        super().__init__(samples)

    def compute_distance(self, i: int, j: int):
        """计算两个文本之间的距离（1-Jarccard相似度），距离越小表示越相近。

        :param s1:文本s1
        :param s2:文本s2
        :return:
        """
        s1, s2 = self.samples[i], self.samples[j]

        return 1 - len(set(s1) & set(s2)) / len(set(s1) | set(s2))
