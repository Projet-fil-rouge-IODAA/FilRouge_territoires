# coding=utf-8
import random as rd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.cluster import AgglomerativeClustering


class CollaborativeClustering():
    '''''
    Ensemble de modèles de collaborative clustering
    '''''

    def __init__(self, *args, clusters = [], modele = AgglomerativeClustering, n_clusters = 4):
        self.pixels = args
        self.modele = modele(n_clusters)
        self.clusters = clusters
        self.n_bandes = len(self.pixels)
        self.n_clusters = n_clusters

    def initial_clustering(self):
        '''''
        format des pixels en entrée, pour chaque bande:
        [[      ] série temporelle pixel 1
         [      ] série temporelle pixel 2
           ...
         [      ] série temporelle pixel 54
        ]
        '''''
        
        for i in range(self.n_bandes):
            self.modele.fit(self.pixels[i])
            self.clusters.append(self.modele.labels_.tolist())

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
                matrice[(4*j)+self.clusters[1+j][i]][self.clusters[0][i]].append(i)
                # matrice[4+self.clusters[2][i]][self.clusters[0][i]].append(i)

        # Matrice de confusion
        confusion = [[len(matrice[j][i]) for i in range(self.n_clusters)] for j in range(self.n_clusters*(self.n_bandes-1))]
        confusion = np.array(confusion).astype(dtype=np.float16)

        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                confusion[i, j] = confusion[i, j]/(max(confusion[i, :].sum(), confusion[:, j].sum())/(self.n_bandes-1))

        # Construction du vecteur clusters
        vect_cluster = [0]*(self.n_clusters*(self.n_bandes-1))
        for i in range(confusion.shape[0]):
            vect_cluster[i] = np.argmax(confusion[i,:])

        # To relabel clustering results
        clusters_relabeled = list(self.clusters)
        j = 1
        while j < self.n_bandes:
            for k in range(0, self.n_clusters):
                clusters_relabeled[j] = np.where(self.clusters[j] == k, vect_cluster[k+4*(j-1)], clusters_relabeled[j])
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