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

# Artificial import of results comming from The notebook 16.0
t2fresults = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 4, 0, 0, 0, 4, 4,
       4, 4, 4, 0, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
       2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0,
       4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 3,
       3, 3, 3, 3, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 4, 0,
       0, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3,
       3, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 4, 0, 0, 0, 0,
       3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
       0, 4, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 4, 4,
       4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
       3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0,
       0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 0, 0, 3, 3,
       3, 3, 3, 3, 3, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4,
       4, 4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 4, 0, 4, 4, 4, 4, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0,
       0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 0, 3, 3, 3, 0, 0, 3, 3, 0, 4, 4, 4,
       4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
       3, 3, 3, 3, 3, 0, 0, 0, 4, 4, 4, 4, 0, 0, 4, 4, 0, 3, 3, 0, 0, 0,
       0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 4, 4, 4, 4, 0, 0, 4, 0,
       0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4,
       4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
       3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0,
       0, 0, 0, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
       2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4,
       4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4,
       4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 3, 3, 0, 0, 0, 0, 0, 3, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 0, 4,
       4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 3, 3, 0, 0,
       0, 0, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0,
       0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0,
       0, 3, 3, 3, 3, 0, 0, 0, 3, 3])

# file = open('/home/julian/FilRouge_territoires/data/processed/pixels_de_interet_dic.pkl', 'rb')
# dic_de_pixels = pickle.load(file)
# file.close()

file = open('/home/julian/FilRouge_territoires/data/processed/pixels_de_interet_list.pkl', 'rb')
pixels_de_interet = pickle.load(file)
file.close()

PATH_IMAGE = "/home/julian/FilRouge_territoires/data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"

results = afficheur_de_resultats(PATH_IMAGE, t2fresults, pixels_de_interet)
results.create_image('results/just_a_test.png', cbar=True)