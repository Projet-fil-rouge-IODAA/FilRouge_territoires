# Nada et Julian 06/02/2024
import random as rd
import numpy as np
import random as rd


class CollaborativeClustering():
    def __init__(self, *args):
        self.clusters = args

    def fit_predict(self):
        n_clusters = []  # on stocke le nombre de clusters pour chaque méthode. len(n_clusters) = nb de méthodes du collaborative clustering
        for i in range(len(self.clusters)):
            n_clusters.append(len(set(self.clusters[i])))

        matrice = [[[] for _ in range(n_clusters[0])] for _ in range(sum(n_clusters[1:]))]

        for i in range(n_clusters[0]):
            matrice[self.clusters[1][i]][self.clusters[0][i]].append(i)
            j = 1
            while j < len(n_clusters):
                matrice[sum(n_clusters[1:j])+self.clusters[1][i]][self.clusters[0][i]].append(i)
                j += 1
                
        # Matrice de confusion
        confusion = [[len(matrice[j][i]) for i in range(n_clusters[0])] for j in range(sum(n_clusters[1:]))]
        confusion = np.array(confusion).astype(dtype=np.float16)
        
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                confusion[i, j] = confusion[i, j]/(max(confusion[i, :].sum(), (confusion[:, j].sum()/(max(n_clusters[1:])-1))))  # max(n_vert,n_bleu) ou n_rouge??

        # Construction du vecteur clusters
        vect_cluster = [0]*(sum(n_clusters[1:]))
        for i in range(confusion.shape[0]):
            vect_cluster[i] = np.argmax(confusion[i,:])
        
        # to relabel clustering results
        clusters_relabeled = self.clusters
        clusters_relabeled[1] = np.where(self.clusters[j] == k, vect_cluster[k], clusters_relabeled[j])
        j = 2
        while j<len(n_clusters):
            for k in range(0, n_clusters[j]):
                clusters_relabeled[j] = np.where(self.clusters[j] == k, vect_cluster[k+sum(n_clusters[1:j-1])], clusters_relabeled[j])
            j += 1
        
        # Matrice de clusterings "relabeled"
        results = np.column_stack(clusters_relabeled)  # attention : cases NaN si les listes clusters ne sont pas de la même taille. np.unique va les ignorer.

        # Vote et génération des clusters finaux
        final_clusters = []
        for i in range(len(self.clusters[0])):
            values, counts = np.unique(results[i, :], return_counts=True)
            if np.all(counts == 1):
                final_clusters.append(rd.randint(0, n_clusters[0]))
            else:
                final_clusters.append(values[counts.argmax()])
        self.final_clusters = final_clusters

        return self.final_clusters
