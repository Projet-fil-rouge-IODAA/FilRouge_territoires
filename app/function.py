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


def our_t2f(model_type, transform_type, data_cube, nombre_clusters, labels, batch_size, n_cores):

    # Imputs zone
    context = {'model_type': model_type, 'transform_type': transform_type}

    # Feature Extraction
    df_feats_i = feature_extraction(data_cube, batch_size=batch_size, p=n_cores)
    print(f'End of feature extraction: {df_feats_i.shape}')

    # Feature Selection
    top_feats = feature_selection(df_feats_i, labels=labels, context=context)
    df_feats = df_feats_i[top_feats]
    print(f'End of feature selection: {df_feats.shape}')

    # Estimating the number of clusters
    if nombre_clusters == 'auto':
        n_clusters = fn(data=df_feats, model_type=model_type, k_min=2, k_max=30,
                        plot_elbow=False)
        print(f'Estimated number of clusters: {n_clusters}')
    else:
        n_clusters = nombre_clusters
        print(f'You have choised: {n_clusters} clusters')

    # Clustering
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type,
                           transform_type=transform_type)
    yhat = model.fit_predict(df_feats)
    print(f'End of clustering: {yhat.shape}')
    return yhat

def t2f_aply(coords, path_image, model_type, transform_type):

    os.system("make create_data")

    COORDS = (255, 82, 305, 102)
    # TODO calculate CUBE_SHAPE first dimension
    # CUBE_SHAPE = (, 141, 8)
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