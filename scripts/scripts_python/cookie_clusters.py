'''
Cookie_tools.py est un fichier qui garde les fonctions nécessaires
à faire plusieurs operations spécifiques du projet fil_rouge.
'''
# TODO: docstrings documentation for all functions.
# Importation des librairies
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import calinski_harabasz_score
import fastdtw as dtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine


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

    dic = {'pix_foret': pix_foret, 'pix_lac': pix_lac, 'pix_apt': pix_apt,
           'pix_ensta': pix_ensta, 'pix_agri': pix_agri, 'pix_danone': pix_danone}
    list = pix_foret + pix_lac + pix_apt + pix_ensta + pix_agri + pix_danone

    return list, dic

class evaluator_de_experiences(object):
    '''
    Classe qui permet d'evaluer les resultats des differents approches
    de clustering.
    '''
    def __init__(self, yhat, pix_list, pix_dic, matrice) -> None:
        self.pix_list = pix_list
        self.pix_dic = pix_dic
        self.yhat = yhat
        self.matrice = matrice

        classes = list()
        for i in self.pix_list:
            for j, k in zip(list(self.pix_dic.keys()), range(0, len(list(self.pix_dic.keys())))):
                if i in list(self.pix_dic[j]):
                    classes.append(k)
        yhat = pd.DataFrame(yhat)
        yhat_double = pd.DataFrame()
        yhat_double['cluster'] = yhat
        yhat_double['class'] = classes
        yhat_double = yhat_double[yhat_double['cluster'] >= 0]
        max_class = yhat_double.groupby('cluster').agg(lambda x:x.value_counts().index[0])
        new_yhat = pd.DataFrame()
        new_yhat['cluster'] = yhat_double['cluster'].map(max_class['class'])

        self.y_hat_clas = new_yhat.to_numpy().squeeze()
        self.y_reel = yhat_double['class'].to_numpy().squeeze()

    def show_list(self):
        '''
        Fonction qui affiche les clusters et les pixels qui les composent.
        '''
        dico = {}
        for cluster in set(self.yhat):
            pixels_in_cluster = np.array(self.pix_list)[self.yhat==cluster].tolist()
            tmp = np.array(self.y_reel)[self.yhat==cluster].tolist()
            ett_of_pixels = [list(self.pix_dic.keys())[i] for i in tmp]
            dico[str(cluster)] = (pixels_in_cluster, ett_of_pixels)
        for key in dico:
            print(f'cluster numero {key}:\n-------------------------------')
            for pixel, ettiquete in zip(dico[key][0], dico[key][1]):
                print(f'{pixel} : {ettiquete}')
            print('-------------------------------')

    def metrics_classif(self):
        '''
        Fonction qui permet de calculer les metrics de classification.
        '''
        return pd.DataFrame({'accuracy': [accuracy_score(self.y_reel, self.y_hat_clas)],
                             'f1_score': [f1_score(self.y_reel, self.y_hat_clas, average = 'macro')]})

    def metrics_clustering(self):
        '''
        Fonction qui permet de calculer les metrics de clustering.
        '''
        return pd.DataFrame({'calinski_harabasz_score':
                             [calinski_harabasz_score(self.matrice, self.yhat)]})

    def confusion_matrix(self):
        '''
        Fonction qui permet de calculer la matrice de confusion.
        '''
        cm = confusion_matrix(self.y_reel, self.y_hat_clas)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.pix_dic.keys()))
        # plot the confusion matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        cmd.plot(colorbar=False, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xticks(range(len(list(self.pix_dic.keys()))))
        ax.set_xticklabels(list(self.pix_dic.keys()), rotation=45)
        plt.plot()
