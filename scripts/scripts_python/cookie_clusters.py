'''
Cookie_tools.py est un fichier qui garde les fonctions nécessaires
à faire plusieurs operations spécifiques du projet fil_rouge.
'''
from scipy.spatial import distance
import collections
import numpy as np
import math

# mesure de similarité dtw entre deux séries temporelles.
# x = un pixel sur un intervalle de temps.


def create_dic_pixels():
    '''
    Fonction qui permet de creer un dictionnaire avec les pixels de
    chaque type d'evolution.
    '''
    pix_foret = [[472, 570], [474, 570], [476, 570], [478, 570], [480, 570],
                [482, 570], [484, 570], [486, 570], [488, 570]]  # ca change uniformement (saison)
    pix_lac = [[392, 567], [392, 580], [401, 577], [401, 567], [395, 570], 
            [395, 576], [397, 571], [394, 598], [388, 532]]  # ca change un peu
    pix_apt = [[405, 448], [408, 444], [412, 446], [412, 463], [407, 465],
            [405, 455], [414, 440], [420, 458], [401, 446]]  # ca change (construction)
    pix_ensta = [[447, 618], [454, 627], [454, 631], [457, 632], [459, 625],
                [450, 641], [443, 636], [439, 629], [433, 617]]  # ca change (construction)
    pix_agri = [[318, 438], [322, 435], [324, 433], [329, 429], [333, 426],
                [337, 424], [339, 422], [344, 418], [350, 414]]  # ca peut changer (saison, plantation)
    pix_danone = [[383, 497], [383, 500], [387, 501], [383, 504], [387, 505],
                [384, 508], [388, 509], [384, 504], [386, 504]]  # ca ne change pas

    return {'pix_foret':pix_foret, 'pix_lac':pix_lac, 'pix_apt':pix_apt,  
            'pix_ensta':pix_ensta, 'pix_agri':pix_agri, 'pix_danone':pix_danone}


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

class evaluator_de_experiences:
    '''
    Classe qui permet d'evaluer les resultats des differents approches
    de clustering.
    '''
    def __init__(self, pixels_de_interet, yhat) -> None:
           self.pixels = pixels_de_interet
           self.yhat = yhat

    def evaluer(yhat,pix_interet):
        '''
        Fonction qui affiche les clusters et les pixels qui les composent.
        Ainsi comme une matrice de confusion et certaines metriques permettant
        de evaluer la qualité du clustering. Et la qualité de la classification.
        '''

        name = ''
        dico = collections.Counter(yhat)
        for key in list(dico.keys()):
            dico[key] = [f'number of vectors = {dico[key]}'] 
            for index,pos in zip(yhat,range(len(yhat))):
                if index == key:
                    if 0<=pos<=8: name = 'pix_danone'
                    elif 9<=pos<=17: name = 'pix_agri'
                    elif 18<=pos<=26: name = 'pix_ensta'
                    elif 27<=pos<=35: name = 'pix_apt'
                    elif 36<=pos<=44: name = 'pix_lac'
                    elif 45<=pos<=53: name = 'pix_foret'

                    dico[key].append(f'{pix_interet[pos]}:{name}')

        for key in dico:
            print(f'cluster numero {key}:\n-------------------------------')
            for part in dico[key]:
                print(f'{part}')
            print('-------------------------------')
