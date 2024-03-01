import sys
sys.path.append("/home/julian/FilRouge_territoires")
import os
from notebooks.cookie_clusters import *
from notebooks.cookie_clusters import find_num_clusters as fn
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
from models.t2f_entire_implementation import our_t2f, save_results
import numpy as np
import pickle
import time


def t2f_apply(coords, model_type, transform_type):

    os.system("make reset_data")
    os.system(f"python scr/features/crop_images.py --coord {coords[0]} {coords[1]} {coords[2]} {coords[3]}")
    os.system("python scr/features/build_features.py.py")

    COORDS = coords
    CUBE_SHAPE = ((coords[2]-coords[0])*(coords[3]-coords[1]), 141, 8)
    PATH_IMAGE = "data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
    name_experiment = f'results/t2f/experience_page_web'

    start=time.time()

    file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
    pixels_de_interet = pickle.load(file)
    file.close()

    data_cube = np.loadtxt(f'data/processed/{CUBE_SHAPE}.csv', delimiter=",").astype(np.float32)
    data_cube = data_cube.reshape(CUBE_SHAPE)

    yhat=our_t2f(model_type=model_type, transform_type=transform_type,
                 data_cube=data_cube, nombre_clusters='auto', labels={},
                 batch_size=100, n_cores=6)

    results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
    results.create_image(name_experiment, cbar=False, axes=False)

    end = time.time()
    print(f"Execution time: {end - start} seconds")

    save_results(f'{name_experiment}.txt', 't2f', yhat,
                 COORDS,
                 model_type, transform_type, end-start)