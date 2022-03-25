'''Homework1: implement KMeans using numpy and compute within-cluster distances
'''

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class MyKMeans(object):
    '''
    Args:
        data (ndarray): input data, it should be of shape (n_samples, n_features).
        n_clusters (int): number of clusters.
    Keyword Args:
        n_init (int): number of times to repeat clustering from different center seeds.
        max_iter (int): maximum number of iterations in each repeat.
        tol (float): absolute tolerance to check convergence.
    '''
    def __init__(self, data, n_clusters, n_init=50, max_iter=300, tol=1e-8):

        self.data = np.asarray(data)
        self.n_samples = self.data.shape[0]
        if n_clusters < 0 or n_clusters > self.n_samples:
            raise Exception("<n_clusters> set wrong.")

        self.n_clusters = n_clusters
        self.n_init = max(1, n_init)
        self.max_iter = max_iter
        self.tol = tol

    def initialize(self):
        indices = np.random.choice(self.n_samples, size=self.n_clusters, replace=False)
        return self.data[indices]

    def allocate(self, centers, samples):
        dists = np.sum((centers[:, None, :] - samples)**2, axis=-1)
        return np.argmin(dists, axis=0)

    def __cluster__(self):
        # random initialize
        centers = self.initialize()
        iter = 0
        while True:
            # allocation phase
            label = self.allocate(centers, self.data)
            # center update phase
            changed = False
            for ii in range(self.n_clusters):
                samplesii = self.data[label == ii]
                if len(samplesii) > 0:
                    new_cii = samplesii.mean(axis=0)
                    if not np.allclose(new_cii, centers[ii], atol=self.tol):
                        changed = True
                    centers[ii] = new_cii

            iter += 1
            if iter >= self.max_iter or (not changed):
                break

        label = self.allocate(centers, self.data)

        return centers, label

    def cluster(self):
        # repeat n_init times to help find global optimal solution
        centers, labels = [], []
        in_cluster_dists = np.zeros([self.n_init, self.n_clusters])
        for ii in range(self.n_init):
            centerii, labelii = self.__cluster__()
            centers.append(centerii)
            labels.append(labelii)
            for jj in range(self.n_clusters):
                samplesjj = self.data[labelii == jj]
                if len(samplesjj) > 0:
                    in_cluster_dists[ii, jj] = \
                        np.linalg.norm(samplesjj - centerii[jj], axis=-1).sum()

        idx = np.argmin(in_cluster_dists.sum(axis=1))
        self.cluster_centers_ = centers[idx]
        self.labels = labels[idx]
        # within-cluster distances
        self.in_cluster_dists = in_cluster_dists[idx]

        return


if __name__ == '__main__':

    # create some data
    x, labels = make_blobs(n_samples = 100, n_features = 2, centers = 10)

    # do kmeans
    kmeans = MyKMeans(x, 10)
    kmeans.cluster()

    print('Within-cluster distances:')
    print(kmeans.in_cluster_dists)

    # use sklearn kmeans
    from sklearn.cluster import KMeans
    sk = KMeans(n_clusters=10)
    sk.fit_predict(x)

    # plot
    figure = plt.figure()

    ax = figure.add_subplot(1,2,1)
    ax.scatter(x[:, 0], x[:, 1], s=2, c='k')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=6,
            color='r')
    ax.set_aspect('equal')
    ax.set_title('MyKMeans')

    ax = figure.add_subplot(1,2,2)
    ax.scatter(x[:, 0], x[:, 1], s=1, c='k')
    ax.scatter(sk.cluster_centers_[:, 0], sk.cluster_centers_[:, 1], s=6,
            color='r')
    ax.set_aspect('equal')
    ax.set_title('sklean kmeans')

    figure.show()
