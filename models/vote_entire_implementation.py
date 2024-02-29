import time
import sys
sys.path.append("/home/mverlynde/FilRouge_territoires")
from notebooks.cookie_clusters import *
from multiple_cluster_combination_system import CollaborativeClustering
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

start=time.time()

COORDS = (255, 82, 305, 102)
CUBE_SHAPE = (1000, 141, 8)
PATH_IMAGE = "data/cropped/cropped_with_texture_crop_SENTINEL2B_20231007-105728-915_L2A_T31UDQ_C_V3-1.tif"
matrice_nir = pd.read_csv('data/processed/vec_nir.csv').to_numpy()
matrice_rouge = pd.read_csv('data/processed/vec_red.csv').to_numpy()
matrice_vert = pd.read_csv('data/processed/vec_green.csv').to_numpy()
matrice_bleu = pd.read_csv('data/processed/vec_blue.csv').to_numpy()
matrice_ndvi = pd.read_csv('data/processed/vec_ndvi.csv').to_numpy()
matrice_ndwi = pd.read_csv('data/processed/vec_ndwi.csv').to_numpy()
matrice_energy = pd.read_csv('data/processed/vec_energy.csv').to_numpy()
matrice_homo = pd.read_csv('data/processed/vec_homogeneity.csv').to_numpy()
name_experiment = f'results/collabclustering/1000_pixels_{COORDS[0]}_{COORDS[1]}_collab_4_clusters'
file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
pixels_de_interet = pickle.load(file)
file.close()

vote_dtw = CollaborativeClustering(matrice_rouge, matrice_nir, matrice_vert, matrice_bleu, matrice_energy, matrice_ndvi, matrice_ndwi, matrice_homo, n_clusters=4)

clusters = vote_dtw.umap_hdbscan_clustering()
yhat = vote_dtw.iccm()

results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
results.create_image(name_experiment, cbar=False, axes=False)

end = time.time()
print(f"Execution time: {end - start} seconds")

save_results_vote(f'{name_experiment}.txt', yhat, COORDS,
                 'dtw_clustering', end-start)



