# Taken from https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
# Just added plotting for 3-k cases

import numpy as np
import random
import matplotlib.pyplot as plt


def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)


def change_coords(array):
    return list(map(list, zip(*array)))


def parse_output(data):
    clusters = data[1]
    points1 = change_coords(clusters[0])
    plt.plot(points1[0], points1[1], 'ro')
    points2 = change_coords(clusters[1])
    plt.plot(points2[0], points2[1], 'g^')
    points3 = change_coords(clusters[2])
    plt.plot(points3[0], points3[1], 'ys')
    centroids = change_coords(data[0])
    plt.plot(centroids[0], centroids[1], 'kx')
    plt.axis([0, 2, 0, 2])
    plt.savefig('kmeans.png', dpi=100, bbox_inches='tight')
    plt.show()


# data = init_board(15)
# data = np.array([[0.5,0.5],[0.6,0.6],[-0.5,-0.5], [-0.6,-0.6],[0.124,0.124], [0.135,0.135]])
data = np.array([[0.1, 0.1], [0.15, 0.15], [0.2, 0.2], [0.3, 0.3], [0.25, 0.25], [0.7, 0.7], [0.75, 0.75], [0.9, 0.9],
                [0.85, 0.85], [1.5, 1.5], [1.55, 1.55], [1.6, 1.6], [1.65, 1.65], [1.7, 1.7]])
print(data)
print(type(data))
out = find_centers(list(data), 3)
parse_output(out)
