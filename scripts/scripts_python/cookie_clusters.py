'''
Cookie_tools.py est un fichier qui garde les fonctions nécessaires
à faire plusieurs operations spécifiques du projet fil_rouge.
'''
from scipy.spatial import distance
import numpy as np
import math

# mesure de similarité dtw entre deux séries temporelles.
# x = un pixel sur un intervalle de temps.


def dtw(x, x_prime):
    '''
    Cette fonction mesure la distance dwt entre deux séries temporelles.
    '''
    R = np.zeros((len(x), len(x_prime)))
    for i in range(len(x)):
        for j in range(len(x_prime)):
            R[i, j] = distance.euclidean(x[i], x_prime[j]) ** 2
            if i > 0 or j > 0:
                R[i, j] += min(
                    R[i-1, j] if i > 0 else math.inf,
                    R[i, j-1] if j > 0 else math.inf,
                    R[i-1, j-1] if (i > 0 and j > 0) else math.inf
                )

    return R[-1, -1] ** (1/2)


def dtw_matrice(x, centroides):
    '''
    Fonction que généralise la fonction "dwt" á l’échelle de la matrice.
    Retourne la matrice de distances entre tous les elements de la 
    matrice de données.
    '''
    distances = np.zeros((len(x), len(centroides)))
    index_i = 0
    index_j = 0
    for i in x:
        for j in centroides:
            distances[index_i, index_j] = dtw(
                i.reshape(-1, 1), j.reshape(-1, 1))
            index_j += 1
        index_i += 1
        index_j = 0
    return distances


def kmeans_dtw(x, k, no_of_iterations):
    '''
    K-means qu'utilise comme fonction de distance le dynamic time warping 
    '''
    idx = np.random.choice(len(x), k, replace=False)
    # Randomly choosing Centroids
    centroids = x[idx, :]  # Step 1

    # finding the distance between centroids and all the data points
    distances = dtw_matrice(x, centroids)  # Step 2

    # Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances])  # Step 3

    # Repeating the above steps for a defined number of iterations
    # Step 4
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Updated Centroids

        distances = distance.cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points
