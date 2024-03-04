import sys
sys.path.append("/home/mverlynde/FilRouge_territoires")
import os
from notebooks.cookie_clusters import *
from notebooks.cookie_clusters import find_num_clusters as fn
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
from models.t2f_entire_implementation import our_t2f, save_results
from models.vote_entire_implementation import *
import numpy as np
import pickle
import time
from subprocess import call


def t2f_apply(coords, model_type, transform_type, number_clusters, nombre_coeurs):

    call("make reset_data", shell=True)
    call(f"python scr/features/crop_images.py --coord {coords[0]} {coords[1]} {coords[2]} {coords[3]}", shell=True)
    call("python scr/features/build_features.py", shell=True)

    COORDS = coords
    CUBE_SHAPE = ((coords[2]-coords[0])*(coords[3]-coords[1]), 141, 8)
    PATH_IMAGE = "data/cropped/"+str(COORDS)+"/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
    os.makedirs('results/t2f', exist_ok=True)
    name_experiment = f'results/t2f/1000_pixels_{COORDS[0]}_{COORDS[1]}_collab_{number_clusters}_clust'

    start=time.time()

    file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
    pixels_de_interet = pickle.load(file)
    file.close()

    data_cube = np.loadtxt(f'data/processed/{CUBE_SHAPE}.csv', delimiter=",").astype(np.float32)
    data_cube = data_cube.reshape(CUBE_SHAPE)

    yhat=our_t2f(model_type=model_type, transform_type=transform_type,
                 data_cube=data_cube, nombre_clusters=number_clusters, labels={},
                 batch_size=100, n_cores=nombre_coeurs)

    results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
    results.create_image(name_experiment, cbar=False, axes=False)

    end = time.time()
    print(f"Execution time: {end - start} seconds")

    save_results(f'{name_experiment}.txt', 't2f', yhat,
                 COORDS,
                 model_type, transform_type, end-start)
    return (name_experiment)


def collabclust_apply(coords, model_type, number_clusters):

    # add the coords to the crop function
    call("make reset_data", shell=True)
    call(f"python scr/features/crop_images.py --coord {coords[0]} {coords[1]} {coords[2]} {coords[3]}", shell=True)
    call("python scr/features/build_features.py", shell=True)

    COORDS = coords
    PATH_IMAGE = "data/cropped/"+str(COORDS)+"/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"

    start=time.time()
    if os.path.exists('data/processed'):
        matrice_nir = pd.read_csv('data/processed/vec_nir.csv').to_numpy()
        matrice_rouge = pd.read_csv('data/processed/vec_red.csv').to_numpy()
        matrice_vert = pd.read_csv('data/processed/vec_green.csv').to_numpy()
        matrice_bleu = pd.read_csv('data/processed/vec_blue.csv').to_numpy()
        matrice_ndvi = pd.read_csv('data/processed/vec_ndvi.csv').to_numpy()
        matrice_ndwi = pd.read_csv('data/processed/vec_ndwi.csv').to_numpy()
        matrice_energy = pd.read_csv('data/processed/vec_energy.csv').to_numpy()
        matrice_homo = pd.read_csv('data/processed/vec_homogeneity.csv').to_numpy()
        os.makedirs('results/collabclustering', exist_ok=True)
        name_experiment = f'results/collabclustering/1000_pixels_{COORDS[0]}_{COORDS[1]}_collab_{number_clusters}_clust'
        
        file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
        pixels_de_interet = pickle.load(file)
        file.close()

        yhat = our_collabclust(model_type=model_type, nombre_clusters=number_clusters,
                    matrice_nir=matrice_nir, 
                    matrice_rouge=matrice_rouge, 
                    matrice_vert=matrice_vert, 
                    matrice_bleu=matrice_bleu, 
                    matrice_ndvi=matrice_ndvi, 
                    matrice_ndwi=matrice_ndwi, 
                    matrice_energy=matrice_energy, 
                    matrice_homo=matrice_homo)

        results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
        results.create_image(name_experiment, cbar=False, axes=False)

        end = time.time()
        print(f"Execution time: {end - start} seconds")

        save_results_vote(f'{name_experiment}.txt', yhat, coords,
                    model_type, end-start)
        
    return (name_experiment)

# t2f_apply(coords= (534, 480, 584, 500), model_type = 'Hierarchical', transform_type='minmax')