# coding=utf-8
import random as rd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import hdbscan
from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)
import fastdtw as dtw
from scipy.spatial.distance import euclidean
import umap

def matrice_dtw(X, distance):

  return R

class CollaborativeClustering():
    '''''
    Ensemble de modèles de collaborative clustering
    '''''

    def __init__(self, *args, clusters = [], n_clusters = 4):
        self.pixels = args
        self.clusters = clusters
        self.n_bandes = len(self.pixels)
        self.n_clusters = n_clusters

    def dtw_clustering(self, modele = KMedoids):
        '''''
        format des pixels en entrée, pour chaque bande:
        [[      ] série temporelle pixel 1
         [      ] série temporelle pixel 2
           ...
         [      ] série temporelle pixel 54
        ]
        '''''
        self.modele = KMedoids(n_clusters=self.n_clusters, metric='precomputed')

        for i in range(self.n_bandes):
            #calcul de la matrice de similarité
            N=self.pixels[i].shape[0]
            R=np.zeros((N,N))
            for j in range(N):
                for k in range(N):
                    R[j, k]= dtw.fastdtw(self.pixels[i][j,:].reshape(-1,1),self.pixels[i][k,:].reshape(-1,1),dist=euclidean)[0]
            #clustering
            self.modele.fit(R)
            self.clusters.append(self.modele.labels_.tolist())
        return self.clusters

    def umap_hdbscan_clustering(self):
        for i in range(self.n_bandes):
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(self.pixels[i])
            hdbscan = HDBSCAN_flat(embedding, cluster_selection_method='eom', n_clusters = self.n_clusters)
            self.clusters.append(hdbscan.labels_.tolist())
        return self.clusters

    def iccm(self):
        '''''
        Iterative Combining Clusterings Method (ICCM)
        Modèle de collaborative clustering qui prend en entrée la liste des clusters pour chaque méthode qui vote.
        Le modèle renvoie une unique liste de clusters. 
        '''''
        matrice = [[[] for _ in range(0, self.n_clusters)] for _ in range(self.n_clusters*(self.n_bandes-1))]

        for i in range(len(self.clusters[0])):
            for j in range(self.n_bandes-1):
                matrice[(self.n_clusters*j)+self.clusters[1+j][i]][self.clusters[0][i]].append(i)

        # for i in range(len(clusters[0])):
    #     matrice[clusters[1][i]][clusters[0][i]].append(i)
    #     matrice[4+clusters[2][i]][clusters[0][i]].append(i)
                
        # Matrice de confusion
        confusion = [[len(matrice[j][i]) for i in range(self.n_clusters)] for j in range(self.n_clusters*(self.n_bandes-1))]
        confusion = np.array(confusion).astype(dtype=np.float16)

        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                confusion[i, j] = confusion[i, j]/(max(confusion[i, :].sum(), confusion[:, j].sum())/(self.n_clusters-1)) # self.n_clusters???

        # Construction du vecteur clusters
        vect_cluster = [0]*(self.n_clusters*(self.n_bandes-1))
        for i in range(confusion.shape[0]):
            vect_cluster[i] = np.argmax(confusion[i,:])

        # To relabel clustering results
        clusters_relabeled = list(self.clusters)
        j = 1
        while j < self.n_bandes:
            for k in range(0, self.n_clusters):
                clusters_relabeled[j] = np.where(self.clusters[j] == k, vect_cluster[k+self.n_clusters*(j-1)], clusters_relabeled[j])
            j += 1

        # Matrice de clusterings "relabeled"
        results = np.column_stack(clusters_relabeled)  # attention : cases NaN si les listes clusters ne sont pas de la même taille. np.unique va les ignorer.

        # Vote et génération des clusters finaux
        final_clusters = []
        for i in range(len(self.clusters[0])):
            values, counts = np.unique(results[i, :], return_counts=True)
            if np.all(counts == 1):
                final_clusters.append(rd.randint(0, self.n_clusters))
            else:
                final_clusters.append(values[counts.argmax()])
      
        return np.array(final_clusters)