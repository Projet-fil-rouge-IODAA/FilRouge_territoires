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


def only_feature_extraction(data_cube, batch_size, coords, n_cores):

    df_feats_i = feature_extraction(data_cube, batch_size=batch_size, p=n_cores)
    print(f'End of feature extraction: {df_feats_i.shape}')
    # save df_feats_i in a csv
    df_feats_i.to_csv(f'results/t2f_inter/{coords[0]}_{coords[1]}.csv', index=False)



def our_t2f_without_extraction(model_type, transform_type, df_feats_i, nombre_clusters, labels):

    # Imputs zone
    context = {'model_type': model_type, 'transform_type': transform_type}

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

##### Doing the entire process ####

# if __name__ == '__main__':

#     COORDS = (255, 82, 305, 102)
#     CUBE_SHAPE = (1000, 141, 8)
#     PATH_IMAGE = "data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
#     model_type = 'KMeans'
#     transform_type = 'minmax'
#     name_experiment = f'results/t2f/1000_pixels_{COORDS[0]}_{COORDS[1]}_t2f_{model_type}_{transform_type}'

#     start=time.time()

#     file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
#     pixels_de_interet = pickle.load(file)
#     file.close()

#     data_cube = np.loadtxt(f'data/processed/{CUBE_SHAPE}.csv', delimiter=",").astype(np.float32)
#     data_cube = data_cube.reshape(CUBE_SHAPE)

#     yhat=our_t2f(model_type=model_type, transform_type=transform_type,
#                  data_cube=data_cube, nombre_clusters='auto', labels={},
#                  batch_size=100, n_cores=6)

#     results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
#     results.create_image(name_experiment, cbar=False, axes=False)

#     end = time.time()
#     print(f"Execution time: {end - start} seconds")

#     save_results(f'{name_experiment}.txt', 't2f', yhat,
#                  COORDS,
#                  model_type, transform_type, end-start)

##### Creating ONLY the csv file with the features ####

# if __name__ == '__main__':

#     COORDS = (255, 82, 305, 102)
#     CUBE_SHAPE = (1000, 141, 8)

#     start=time.time()

#     data_cube = np.loadtxt(f'data/processed/{CUBE_SHAPE}.csv', delimiter=",").astype(np.float32)
#     data_cube = data_cube.reshape(CUBE_SHAPE)

#     only_feature_extraction(data_cube, batch_size=100, coords=COORDS, n_cores=6)

#     end = time.time()
#     print(f"Execution time: {end - start} seconds")

##### Doing ONLY the feature selection and clustering ####

# if __name__ == '__main__':

#     for i in ['Hierarchical', 'KMeans']:
#         for j in ['minmax', 'std']:

#             start=time.time()

#             COORDS = (255, 82, 305, 102)
#             CUBE_SHAPE = (1000, 141, 8)
#             PATH_IMAGE = "data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
#             model_type = i
#             transform_type = j
#             df_name = f'results/t2f_inter/{COORDS[0]}_{COORDS[1]}.csv'
#             df = pd.read_csv(df_name)
#             name_experiment = f'results/t2f/1000_pixels_{COORDS[0]}_{COORDS[1]}_t2f_{model_type}_{transform_type}_3_clusters'

#             file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
#             pixels_de_interet = pickle.load(file)
#             file.close()

#             yhat = our_t2f_without_extraction(model_type, transform_type, df_feats_i=df,
#                                             nombre_clusters=3, labels={})

#             results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
#             results.create_image(name_experiment, cbar=False, axes=False)

#             end = time.time()
#             print(f"Execution time: {end - start} seconds")

#             save_results(f'{name_experiment}.txt', 't2f', yhat,
#                         COORDS,
#                         model_type, transform_type, end-start)
