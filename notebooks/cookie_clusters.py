'''
Cookie_tools.py est un fichier qui garde les fonctions nécessaires
à faire plusieurs operations spécifiques du projet fil_rouge.
'''
# TODO: docstrings documentation for all methodes.
# Importation des librairies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import rasterio
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
        plt.xlabel('Predicted_label')
        plt.ylabel('True_label')

        # Display the plot
        plt.show()


class afficheur_de_resultats(object):
    '''
    Classe qui permet d'afficher un vecteur de resultats
    sur une des images satellitaires.
    '''
    def __init__(self, image_path, yhat, pix_list) -> None:
        self.image_path = image_path
        self.yhat = yhat
        self.pix_list = pix_list
        self.class_colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple',
                             'pink', 'brown', 'lime', 'teal', 'olive', 'navy',
                             'maroon', 'aqua', 'fuchsia', 'silver', 'gray',
                             'black', 'indigo', 'coral', 'gold', 'darkgreen',
                             'darkblue', 'darkred', 'darkorange', 'darkviolet',
                             'darkgray', 'lightgray']

    def create_image(self, name_image = 'noName.png', cbar = True):
        '''
        Fonction qui permet d'afficher les clusters sur l'image.
        '''

        src = rasterio.open(self.image_path)
        red = src.read(2)
        green = src.read(3)
        blue = src.read(4)

        redn = (red/6).astype(int)
        greenn = (green/6).astype(int)
        bluen = (blue/6).astype(int)

        # Create RGB natural color composite
        rgb = np.dstack((redn, greenn, bluen))
        # Create the results matrix
        results = np.zeros((rgb.shape[0], rgb.shape[1]))
        for i in range(0, len(self.pix_list)):
            results[self.pix_list[i][0], self.pix_list[i][1]] = self.yhat[i]+1
        # changes 0 to nan
        results[results == 0] = np.nan

        cookie_map = colors.ListedColormap(self.class_colors
                                           [:len(np.unique(self.yhat))])

        # Let's see how our color composite looks like
        # To have high quality images
        # (specially in notebooks for isolated pixels),
        # change the commentent line below.
        if cbar:
            plt.figure(figsize=(100, 100), dpi=100)
            # plt.figure(figsize=(rgb.shape[1]/100, rgb.shape[0]/100), dpi=100)
            ax = plt.gca()
            ax.imshow(rgb, alpha=0.3)
            clusters = ax.imshow(results, cmap=cookie_map)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            plt.colorbar(clusters, cax=cax)
            plt.savefig(f"{name_image}", bbox_inches="tight")

        else:
            plt.figure(figsize=(100, 100), dpi=100)
            # plt.figure(figsize=(rgb.shape[1]/100, rgb.shape[0]/100), dpi=100)
            plt.imshow(rgb, alpha=0.3)
            plt.imshow(results, cmap=cookie_map)
            plt.savefig(f"{name_image}", bbox_inches="tight")

class afficheur_carte_clusters(object):
    '''
    Classe qui permet d'afficher un vecteur de resultats
    sur une des images satellitaires.
    '''
    def __init__(self, image_path, yhat, pix_list) -> None:
        self.image_path = image_path
        self.yhat = yhat
        self.pix_list = pix_list
        self.class_colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple',
                             'pink', 'brown', 'lime', 'teal', 'olive', 'navy',
                             'maroon', 'aqua', 'fuchsia', 'silver', 'gray',
                             'black', 'indigo', 'coral', 'gold', 'darkgreen',
                             'darkblue', 'darkred', 'darkorange', 'darkviolet',
                             'darkgray', 'lightgray']

    def create_image(self, name_image = 'noName.png', cbar = True):
        '''
        Fonction qui permet d'afficher les clusters sur l'image.
        '''

        src = rasterio.open(self.image_path)
        red = src.read(2)
        green = src.read(3)
        blue = src.read(4)

        redn = (red/6).astype(int)
        greenn = (green/6).astype(int)
        bluen = (blue/6).astype(int)

        # Create RGB natural color composite
        rgb = np.dstack((redn, greenn, bluen))
        # Create the results matrix
        results = np.zeros((red.shape[0], red.shape[1]))
        for i in range(0, len(self.pix_list)):
            results[self.pix_list[i][0], self.pix_list[i][1]] = self.yhat[i]+1
        # changes 0 to nan
        # results[results == 0] = np.nan

        cookie_map = colors.ListedColormap(self.class_colors
                                           [:len(np.unique(self.yhat))])

        # Let's see how our color composite looks like
        # To have high quality images
        # (specially in notebooks for isolated pixels),
        # change the commentent line below.
        if cbar:
            # plt.figure(figsize=(100, 100), dpi=100)
            plt.figure(figsize=(rgb.shape[1]/10, rgb.shape[0]/10), dpi=100)
            ax = plt.gca()
            ax.imshow(rgb, alpha=0.7)
            clusters = ax.imshow(results, cmap=cookie_map, alpha=0.3)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            plt.colorbar(clusters, cax=cax)
            plt.savefig(f"{name_image}", bbox_inches="tight")

        else:
            # plt.figure(figsize=(100, 100), dpi=100)
            plt.imshow(rgb, alpha=0.7)
            plt.imshow(results, alpha=0.5, cmap=cookie_map)
            plt.savefig(f"{name_image}", bbox_inches="tight")