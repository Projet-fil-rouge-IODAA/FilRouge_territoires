import time
from notebooks.cookie_clusters import *
from multiple_cluster_combination_system import CollaborativeClustering

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
df_name = f'results/t2f_inter/{COORDS[0]}_{COORDS[1]}.csv'
df = pd.read_csv(df_name)
name_experiment = f'results/collabclustering/1000_pixels_{COORDS[0]}_{COORDS[1]}_collab_3_clusters'
file = open('data/processed/pixels_de_interet_list.pkl', 'rb')
pixels_de_interet = pickle.load(file)
file.close()

vote_dtw = CollaborativeClustering(?imagettes, n_clusters=3)

clusters = vote_dtw.dtw_clustering()
yhat = vote_dtw.iccm()

results = afficheur_de_resultats(PATH_IMAGE, yhat, pixels_de_interet)
results.create_image(name_experiment, cbar=False, axes=False)

end = time.time()
print(f"Execution time: {end - start} seconds")

save_results_vote(f'{name_experiment}.txt', y_hat, COORDS,
                 'dtw_clustering', end-start)



