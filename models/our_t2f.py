import sys
sys.path.append("/home/julian/FilRouge_territoires/notebooks")

from notebooks.cookie_clusters import find_num_clusters
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
import os
import numpy as np
import pickle



class Time2Feature(object):
    '''
    Classe qui permet de piloter une execution complete ou par parties de l'approche
    Time2Feature, qui consiste a extraire des features a partir d'un cube de donnees
    T2F, calcule d'attributs intra instances (using TSFRESH entre bandes) et inter instances
    (using differents mesures de distance entre les pixels).
    
    Parameters
    ----------
    IN_DIR: le dossier d'entree.

    OUT_DIR: le dossier de sortie.

    cube_taille: la taille du cube de donnees.

    model_type: le type de modele a utiliser.

    transform_type: le type de transformation a utiliser.

    n_cores: le nombre de coeurs a utiliser.

    batch_size: la taille des batchs.

    n_clusters: le nombre de clusters a utiliser, si l'utilisateur ne le specifie pas,
        on va utiliser du code pour le trouver.

    labels: un dictionnaire contenant les etiquettes des pixels,
        utile si l'utilisateur veut utiliser un approche semi-supervisee.

    '''
    def __init__(self, IN_DIR, OUT_DIR, cube_taille, model_type,
                 transform_type, n_cores, batch_size, n_clusters=0, labels={}):

        self.IN_DIR = IN_DIR
        self.OUT_DIR = OUT_DIR
        self.cube_taille = cube_taille
        self.model_type = model_type
        self.transform_type = transform_type
        self.n_clusters = n_clusters
        self.n_cores = n_cores
        self.batch_size = batch_size
        self.labels = labels

        # Creating a context dictionary
        self.context = {'model_type': self.model_type,
                        'transform_type': self.transform_type,
                        'labels': self.labels}


        file = open(f'../{self.IN_DIR}/pixels_de_interet_list.pkl', 'rb')
        self.pixels_de_interet = pickle.load(file)
        file.close()

        data_cube = np.loadtxt(f'../{self.IN_DIR}/{self.cube_taille}.csv',
                               delimiter=",")
        self.data_cube = data_cube.reshape(cube_taille)

    def _feature_extraction(self):
        print('Doing feature extraction')
        self.df_feats = feature_extraction(self.data_cube,
                                           batch_size=self.batch_size,
                                           p=self.n_cores)
        print(f'The shape of the dataset after the feature extraction is {self.df_feats.shape}')

    def _feature_selection(self):
        top_feats = feature_selection(self.df_feats, context=self.context)
        self.df_feats = self.df_feats[top_feats]
        print(f'The shape of the dataset after the feature selection is {self.df_feats.shape}')

    def _n_clusters(self):
        if self.n_clusters == 0:
            print('you have not specified the number of clusters.')
            print('Finding the number of clusters...')
            self.n_clusters = find_num_clusters(data=self.df_feats,
                                                model_type=self.context['model_type'],
                                                k_min=2,
                                                k_max=30)
            print(f'We are going to use {self.n_clusters} clusters.')
        else:
            print(f'We are going to use {self.n_clusters} clusters.')

    def _clustering(self):
        model = ClusterWrapper(n_clusters=self.n_clusters,
                               model_type=self.context['model_type'],
                               transform_type=self.context['transform_type'])
        print(f'Executing {self.context["model_type"]} clustering...')
        return model.fit_predict(self.df_feats)

    def entire_t2f(self):
        self._feature_extraction()
        self._feature_selection()
        self._n_clusters()
        yhat = self._clustering()
        print(f'End of T2F')
        return yhat
