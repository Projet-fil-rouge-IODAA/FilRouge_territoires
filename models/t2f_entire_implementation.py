import sys
sys.path.append("/home/julian/FilRouge_territoires")
from notebooks.cookie_clusters import *
from notebooks.cookie_clusters import find_num_clusters as fn
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
import numpy as np
import pickle
import time

# include the time of the execution
start = time.time()

def save_results(out_path, t2f_or_vote, y_hat, coords,
                 cluster_method, transform_type, exec_time):
    '''
    Fonction qui permet de sauvegarder les resultats
    de clustering.
    '''
    # Save the results
    with open(out_path, 'w') as f:
        f.write(f'{t2f_or_vote}\n')
        f.write(f'Number of pixels: {len(y_hat)}\n')
        f.write(f'Execution time: {exec_time} seconds\n')
        f.write(f'Coords: {coords}\n')
        f.write(f'Cluster_method: {cluster_method}\n')
        f.write(f'Transform_type: {transform_type}\n')
        f.write(f'yhat:\n\
        {y_hat}\n')
        f.close()

def our_t2f(model_type, transform_type, data_cube, labels, batch_size, n_cores):

    # Imputs zone
    context = {'model_type': model_type, 'transform_type': transform_type}

    # Feature Extraction
    df_feats_i = feature_extraction(data_cube, batch_size=batch_size, p=n_cores)
    print(f'End of feature extraction: {df_feats_i.shape}')

    # Feature Selection
    top_feats = feature_selection(df_feats_i, labels=labels, context=context)
    df_feats = df_feats_i[top_feats]
    print(f'End of feature selection: {df_feats.shape}')

    # Stimating the number of clusters
    n_clusters = fn(data=df_feats, model_type=model_type, k_min=2, k_max=30,
                    plot_elbow=False)
    print(f'Estimated number of clusters: {n_clusters}')

    # Clustering
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type,
                           transform_type=transform_type)
    yhat = model.fit_predict(df_feats)
    print(f'End of clustering: {yhat.shape}')
    return yhat


if __name__ == '__main__':

    # for i in ['Hierarchical', 'Kmeans']:
    #     for j in ['minmax', 'std']:
    #             CUBE_SHAPE = (1000, 141, 8)
    #             PATH_IMAGE = "data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
    #             model_type = i
    #             transform_type = j
    #             name_experiment = f'results/t2f/1000_pixels_422_399_t2f_{model_type}_{transform_type}'

    #             start=time.time()

    #             file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
    #             pixels_de_interet = pickle.load(file)
    #             file.close()

    #             data_cube = np.loadtxt(f'data/processed/{CUBE_SHAPE}.csv', delimiter=",").astype(np.float32)
    #             data_cube = data_cube.reshape(CUBE_SHAPE)

    #             yhat=our_t2f(model_type=model_type, transform_type=transform_type,
    #                         data_cube=data_cube, labels={},
    #                         batch_size=100, n_cores=6)

    #             results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
    #             results.create_image(name_experiment, cbar=False, axes=False)

    #             end = time.time()
    #             print(f"Execution time: {end - start} seconds")

    #             save_results(f'{name_experiment}.txt', 't2f', yhat,
    #                         (422, 399, 472, 419),
    #                         model_type, transform_type, end-start)

# Other approach whitout cycles:
    CUBE_SHAPE = (1000, 141, 8)
    PATH_IMAGE = "data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
    model_type = 'KMeans'
    transform_type = 'minmax'
    name_experiment = f'results/t2f/1000_pixels_422_399_t2f_{model_type}_{transform_type}'

    start=time.time()

    file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
    pixels_de_interet = pickle.load(file)
    file.close()

    data_cube = np.loadtxt(f'data/processed/{CUBE_SHAPE}.csv', delimiter=",").astype(np.float32)
    data_cube = data_cube.reshape(CUBE_SHAPE)

    yhat=our_t2f(model_type=model_type, transform_type=transform_type,
                 data_cube=data_cube, labels={},
                 batch_size=100, n_cores=6)

    results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
    results.create_image(name_experiment, cbar=False, axes=False)

    end = time.time()
    print(f"Execution time: {end - start} seconds")

    save_results(f'{name_experiment}.txt', 't2f', yhat,
                 (422, 399, 472, 419),
                 model_type, transform_type, end-start)
