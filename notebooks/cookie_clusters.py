'''
Cookie_tools.py est un fichier qui garde les fonctions nécessaires
à faire plusieurs operations spécifiques du projet fil_rouge.
'''
# TODO: docstrings documentation for all methodes.
# Importation des librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import calinski_harabasz_score
import seaborn as sns


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
            for j, k in zip(list(self.pix_dic.keys()),
                            range(0, len(list(self.pix_dic.keys())))):
                if i in list(self.pix_dic[j]):
                    classes.append(k)
        yhat = pd.DataFrame(yhat)
        yhat_double = pd.DataFrame()
        yhat_double['cluster'] = yhat
        yhat_double['class'] = classes
        yhat_double = yhat_double[yhat_double['cluster'] >= 0]
        max_class = (yhat_double.groupby('cluster')
                     .agg(lambda x: x.value_counts().index[0]))
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
            pixels_in_cluster = (np.array(self.pix_list)
                                 [self.yhat == cluster].tolist())
            tmp = np.array(self.y_reel)[self.yhat == cluster].tolist()
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
        return pd.DataFrame({'accuracy': [accuracy_score(self.y_reel,
                                                         self.y_hat_clas)],
                             'f1_score': [f1_score(self.y_reel,
                                                   self.y_hat_clas,
                                                   average='macro')]})

    def metrics_clustering(self):
        '''
        Fonction qui permet de calculer les metrics de clustering.
        '''
        return pd.DataFrame({'calinski_harabasz_score':
                             [calinski_harabasz_score
                              (self.matrice, self.yhat)]})

    def confusion_matrix(self):
        '''
        Fonction qui permet de calculer la matrice de confusion.
        '''
        cm = confusion_matrix(self.y_reel, self.y_hat_clas)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=list(self.pix_dic.keys()))
        # plot the confusion matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        cmd.plot(colorbar=False, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xticks(range(len(list(self.pix_dic.keys()))))
        ax.set_xticklabels(list(self.pix_dic.keys()), rotation=45)
        plt.plot()

    def cluster_distribution(self):
        '''
        Fonction qui permet de calculer la distribution des clusters.
        '''
        unique_strings = np.unique(self.yhat.astype(str))
        dic_dist = {s: {pix: 0 for pix in list(self.pix_dic.keys())}
                    for s in unique_strings}
        for classe, coord in zip(self.yhat.astype(str), self.pix_list):
            for key in dic_dist[classe]:
                if coord in self.pix_dic[key]:
                    dic_dist[classe][key] += 1

        df = pd.DataFrame(dic_dist)
        # Create a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(df, annot=True, fmt='d', cmap='viridis', cbar=False)

        # Label the plot
        plt.title('Cluster Distribution')
        plt.xlabel('Class')
        plt.ylabel('Cluster')

        # Display the plot
        plt.show()
