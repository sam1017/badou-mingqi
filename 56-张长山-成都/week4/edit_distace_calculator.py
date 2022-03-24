import functools
import warnings
from typing import List

import Levenshtein

from distance_calculator import DistanceCalculator


class EditDistanceCalculator(DistanceCalculator):
    def __init__(self, samples: List[str]) -> None:
        super().__init__(samples)

    def compute_distance(self, i: int, j: int):
        s1, s2 = self.samples[i], self.samples[j]
        # return EditDistanceCalculator.compute_edit_distance(s1, s2)
        # return self.compute_levenshtein_distance_dp(s1, s2)
        return self.compute_levenshtein_distance(s1, s2)

    @staticmethod
    @functools.cache
    def compute_edit_distance(s1, s2):
        warnings.warn(
            "this function is time consuming for recursive implement",
            DeprecationWarning,
        )
        m, n = len(s1), len(s2)
        if m == 0:
            return n
        if n == 0:
            return m

        if s1[m - 1] == s2[n - 1]:  # 最后1个字符相等
            return EditDistanceCalculator.compute_edit_distance(
                s1[: m - 1], s2[: n - 1]
            )
        else:
            # d(abc->xyd) = d(ab->xyd) + 1
            d1 = EditDistanceCalculator.compute_edit_distance(s1[: m - 1], s2[:n]) + 1
            # d(abc->xyd) = d(abc->xy) + 1
            d2 = EditDistanceCalculator.compute_edit_distance(s1[:m], s2[: n - 1]) + 1
            # d(abc->xyd) = d(ab->xy) + 1
            d3 = (
                EditDistanceCalculator.compute_edit_distance(s1[: m - 1], s2[: n - 1])
                + 1
            )
            return min(d1, d2, d3)

    def compute_levenshtein_distance(self, s1, s2):
        return Levenshtein.distance(s1, s2)

    def compute_levenshtein_distance_dp(self, s1, s2):
        m, n = len(s1) + 1, len(s2) + 1
        dp = [[0 for _ in range(n)] for _ in range(m)]  # dp[i][j] -> s[:i]->s[:j]的编辑距离

        for i in range(m):
            dp[i][0] = i
        for j in range(n):
            dp[0][j] = j

        for i in range(1, m):
            for j in range(1, n):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    d1 = dp[i][j - 1] + 1
                    d2 = dp[i - 1][j] + 1
                    d3 = dp[i - 1][j - 1] + 1
                    dp[i][j] = min(d1, d2, d3)

        return dp[m - 1][n - 1]


if __name__ == "__main__":
    edc = EditDistanceCalculator(["", "sitting"])
    print(edc.compute_levenshtein_distance_dp("xyzz", "xed"))
