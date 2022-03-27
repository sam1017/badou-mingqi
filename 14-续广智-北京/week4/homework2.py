'''Homework2: implement KMeans on non-vectorized data.
'''
import numpy as np
import random

class MyKMeans(object):
    '''
    Args:
        data (list): input data, each element is one record.
        n_clusters (int): number of clusters.
        dist_func (callable): function to compute inter-sample distance. Should
            return a single scalar value in range [0, 1].
    Keyword Args:
        n_init (int): number of times to repeat clustering from different center seeds.
        max_iter (int): maximum number of iterations in each repeat.
    '''
    def __init__(self, data, n_clusters, dist_func, n_init=20, max_iter=300):

        self.data = data
        self.n_samples = len(data)
        if n_clusters < 0 or n_clusters > self.n_samples:
            raise Exception("<n_clusters> set wrong.")

        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.n_init = max(1, n_init)
        self.max_iter = max_iter

        self.dist_dict = {}  # cache sample distances

    def __dist__(self, ii, jj):
        '''Distance between 2 samples'''
        key = frozenset([ii, jj])  # assume dist(x, y) == dist(y, x)
        if key not in self.dist_dict:
            self.dist_dict[key] = self.dist_func(self.data[ii], self.data[jj])
        return self.dist_dict[key]

    def initialize(self):
        return np.random.choice(self.n_clusters, size=self.n_samples, replace=True)

    def allocate(self, labels, samples):

        new_labels = np.zeros(self.n_samples).astype('int')
        for ii in range(self.n_samples):
            distsii = np.ones(self.n_clusters) * np.inf
            for jj in range(self.n_clusters):
                # mean distance to all samples in cluster jj:
                idxjj = np.where(labels == jj)[0]
                if len(idxjj) > 0:
                    distsii[jj] = np.mean([self.__dist__(ii, kk) for kk in idxjj])
            new_labels[ii] = np.argmin(distsii)

        return new_labels

    def __cluster__(self):
        # random initialize
        labels = self.initialize()
        iter = 0
        while True:
            new_labels = self.allocate(labels, self.data)
            iter += 1
            if iter >= self.max_iter or np.all(labels == new_labels):
                print('Stop at iteration:', iter)
                break

            labels = new_labels

        # compute within-cluster distances: defined as the average of
        # all pair-wise distances
        in_cluster_dists = np.ones(self.n_clusters)
        for ii in range(self.n_clusters):
            idxii = np.where(labels == ii)[0]
            nii = len(idxii)
            if nii > 1:
                distii = 0
                for jj in range(len(idxii) - 1):
                    for kk in range(jj+1, len(idxii)):
                        distii += self.__dist__(idxii[jj], idxii[kk])
                distii /= (nii * (nii - 1))
                in_cluster_dists[ii] = distii

        return labels, in_cluster_dists

    def cluster(self):

        # repeat n_init times to find global optimal solution
        labels = []
        in_cluster_dists = np.zeros([self.n_init, self.n_clusters])

        for ii in range(self.n_init):
            labelii, distii = self.__cluster__()
            labels.append(labelii)
            in_cluster_dists[ii] = distii

        idx = np.argmin(in_cluster_dists.sum(axis=1))
        self.labels = labels[idx]
        # within-cluster distances
        self.in_cluster_dists = in_cluster_dists[idx]

        return


def jaccard(x, y):
    inter = len(set(x).intersection(set(y)))
    union = len(set(x).union(set(y)))
    if union == 0:
        # when both are ''
        return 0
    return 1 - inter/union


if __name__ == '__main__':

    # create some dummy data
    x1 = [''.join(random.choices('abc', k=6)) for _ in range(6)]
    x2 = [''.join(random.choices('xyz', k=6)) for _ in range(6)]
    x3 = [''.join(random.choices('cdx', k=6)) for _ in range(6)]
    x = x1 + x2 + x3
    print('Data to cluster:')
    print(x)

    # do kmeans
    km = MyKMeans(x, 3, jaccard)
    km.cluster()

    print('\nLabels:')
    print(km.labels)
    print('Within-cluster distances:')
    print(km.in_cluster_dists)

    # print some words in each cluster
    idx = np.argsort(km.in_cluster_dists)
    for ii in idx:
        print('\nCluster: %d, mean within-cluster distance: %.2f' %(ii, km.in_cluster_dists[ii]))
        print([x[jj] for jj in np.where(km.labels == ii)[0]])
