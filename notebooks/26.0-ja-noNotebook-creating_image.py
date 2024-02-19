'''
A notebook to create an image from the results of the clustering
The .py native file is used to test the class afficheur_de_resultats
in a different environment.
All this because the default parameters of matplotlib
are not the same as the ones used in the notebook.
'''
import pickle
import numpy as np
from cookie_clusters import afficheur_de_resultats

t2fresults = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5,
                       3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5,
                       5, 5, 3, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 0, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2,
                       2, 2, 2, 0, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 3,
                       3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1])

file = open('/home/julian/FilRouge_territoires/data/processed/ \
            pixels_de_interet_dic.pkl', 'rb')
dic_de_pixels = pickle.load(file)
file.close()

file = open('/home/julian/FilRouge_territoires/data/processed/ \
            pixels_de_interet_list.pkl', 'rb')
pixels_de_interet = pickle.load(file)
file.close()

PATH_IMAGE = "/home/julian/FilRouge_territoires/data/raw/ \
crop_SENTINEL2A_20151226-111142-750_L2A_T31UDQ_D_V1-1.tif"

results = afficheur_de_resultats(PATH_IMAGE, t2fresults, pixels_de_interet)
results.create_image('second_test.png')