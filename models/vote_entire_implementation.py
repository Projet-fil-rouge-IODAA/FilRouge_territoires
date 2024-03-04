import time
import sys
from notebooks.cookie_clusters import *
from .multiple_cluster_combination_system import CollaborativeClustering
import pickle 

def save_results_vote(out_path, y_hat, coords,
                 cluster_method, exec_time):
    '''
    Fonction qui permet de sauvegarder les resultats
    de clustering.
    '''
    # Save the results
    with open(out_path, 'w') as f:
        f.write(f'Cluster method: {cluster_method}\n')
        f.write(f'Number of pixels: {len(y_hat)}\n')
        f.write(f'Execution time: {exec_time} seconds\n')
        f.write(f'Coords: {coords}\n')
        f.write(f'yhat:\n\
        {y_hat}\n')
        f.close()

def our_collabclust(model_type, nombre_clusters,
                    matrice_nir, 
                    matrice_rouge, 
                    matrice_vert, 
                    matrice_bleu, 
                    matrice_ndvi, 
                    matrice_ndwi, 
                    matrice_energy, 
                    matrice_homo):

    vote_dtw = CollaborativeClustering(matrice_rouge, matrice_nir, matrice_vert, matrice_bleu, matrice_energy, matrice_ndvi, matrice_ndwi, matrice_homo, n_clusters=4)

    if model_type == "DTW-Kmedoids":
        clusters = vote_dtw.dtw_clustering()
    else:
        clusters = vote_dtw.umap_hdbscan_clustering()
    yhat = vote_dtw.iccm()
    return yhat

# results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
# results.create_image(name_experiment, cbar=False, axes=False)

# print(f"Execution time: {end - start} seconds")

# save_results_vote(f'{name_experiment}.txt', yhat, COORDS,
#                  'dtw_clustering', end-start)



