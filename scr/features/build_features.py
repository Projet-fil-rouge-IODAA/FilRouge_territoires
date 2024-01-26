import os
import re
import rasterio as rio
import pandas as pd
import numpy as np
import pickle
import shutil

IN_DIR = "data/raw"
OUT_DIR = "data/processed"

# making the output directory
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.mkdir(OUT_DIR)


def create_dic_pixels():
    '''
    Fonction qui permet de creer un dictionnaire avec les pixels de
    chaque type d'evolution.
    '''
    pix_foret = [[472, 570], [474, 570], [476, 570], [478, 570],
                 [480, 570], [482, 570], [484, 570], [486, 570], [488, 570]]
    # ca change uniformement (saison).
    pix_lac = [[392, 567], [392, 580], [401, 577], [401, 567],
               [395, 570], [395, 576], [397, 571], [394, 598], [388, 532]]
    # ca ne change pas.
    pix_apt = [[405, 448], [408, 444], [412, 446], [412, 463],
               [407, 465], [405, 455], [414, 440], [420, 458], [401, 446]]
    # ca change (construction).
    pix_ensta = [[447, 618], [454, 627], [454, 631], [457, 632],
                 [459, 625], [450, 641], [443, 636], [439, 629], [433, 617]]
    # ca change (construction).
    pix_agri = [[318, 438], [322, 435], [324, 433], [329, 429],
                [333, 426], [337, 424], [339, 422], [344, 418], [350, 414]]
    # ca peut changer (saison, plantation).
    pix_danone = [[383, 497], [383, 500], [387, 501], [383, 504],
                  [387, 505], [384, 508], [388, 509], [384, 504], [386, 504]]
    # ca ne change pas.

    dic = {'pix_foret': pix_foret, 'pix_lac': pix_lac,
           'pix_apt': pix_apt, 'pix_ensta': pix_ensta,
           'pix_agri': pix_agri, 'pix_danone': pix_danone}
    list = pix_foret + pix_lac + pix_apt + pix_ensta + pix_agri + pix_danone

    return list, dic


pixels_de_interet, dic_pix = create_dic_pixels()

# Saving the list and the dictionary as pickle objects.
with open(OUT_DIR + '/pixels_de_interet_dic.pkl', 'wb') as file:
    pickle.dump(dic_pix, file)

with open(OUT_DIR + '/pixels_de_interet_list.pkl', 'wb') as file:
    pickle.dump(pixels_de_interet, file)

images_list = os.listdir(IN_DIR)

# put all the images in a list, ready to be read.
images_2A = list()
images_2B = list()
prog = re.compile(r'\w+2A')
for image in images_list:
    if prog.match(image):
        images_2A.append(image)
    else:
        images_2B.append(image)

images_2A = sorted(images_2A, key=lambda date: date[16:24])
images_2B = sorted(images_2B, key=lambda date: date[16:24])
all_images = images_2A + images_2B
all_images = sorted(all_images, key=lambda date: date[16:24])

# Initiate lists to compose the data frame.
date = []
pixel_r = []
pixel_g = []
pixel_b = []
pixel_ir = []
x_coord = []
y_coord = []

# read each image --> select interesting pixels --> add to list
for temp, img in zip(range(len(all_images)), all_images):
    img = IN_DIR + "/" + img
    raster = rio.open(img)
    band_ir = raster.read(1)
    band_r = raster.read(2)
    band_g = raster.read(3)
    band_b = raster.read(4)

    for px_x, px_y in pixels_de_interet:
        ir = band_ir[px_x, px_y]
        r = band_r[px_x, px_y]
        g = band_g[px_x, px_y]
        b = band_b[px_x, px_y]

        date.append(temp)
        x_coord.append(px_x)
        y_coord.append(px_y)
        pixel_ir.append(ir)
        pixel_r.append(r)
        pixel_g.append(g)
        pixel_b.append(b)

# The final DF has the values for all the bands and time periods.
dic = {'date': date,
       'x_coord': x_coord,
       'y_coord': y_coord,
       'pixel_ir': pixel_ir,
       'pixel_r': pixel_r,
       'pixel_g': pixel_g,
       'pixel_b': pixel_b}

data = pd.DataFrame(dic)
data.to_csv(OUT_DIR + "/by_date_and_coord.csv", index=False)

# ------------------ SECOND DFs ---------------------
'''
A second approach to the Data frames construction is to make 2D arrays
each row is a pixel and each column is a time step.
'''
# Careful! During the construction of this Data frames,
# we expect to have zeros at the denominator, we exclude this erros.
# We'll manage the Nan values during the construction.
np.seterr(divide='ignore', invalid='ignore')

line = 0
matrice_rouge = np.zeros((len(pixels_de_interet), len(all_images)))
matrice_vert = np.zeros((len(pixels_de_interet), len(all_images)))
matrice_bleu = np.zeros((len(pixels_de_interet), len(all_images)))
matrice_nir = np.zeros((len(pixels_de_interet), len(all_images)))
matrice_ndvi = np.zeros((len(pixels_de_interet), len(all_images)))
matrice_ndwi = np.zeros((len(pixels_de_interet), len(all_images)))

for image in all_images:
    with rio.open(IN_DIR + '/' + image, 'r') as ds:
        band1 = ds.read(1)  # Near Infrared.
        band2 = ds.read(2)  # Red.
        band3 = ds.read(3)  # Green.
        band4 = ds.read(4)  # Blue.
        # Normalized Difference Vegetation Index.
        ndvi = (band1-band2)/(band1+band2)
        ndvi = np.nan_to_num(ndvi)
        # Normalized Difference Water Index.
        ndwi = (band3-band1)/(band3+band1)
        ndwi = np.nan_to_num(ndwi)
    # Extraction of target pixels.
    for pixel, j in zip(pixels_de_interet, range(len(pixels_de_interet))):
        matrice_vert[j, line] = band3[pixel[0], pixel[1]]
        matrice_rouge[j, line] = band2[pixel[0], pixel[1]]
        matrice_bleu[j, line] = band4[pixel[0], pixel[1]]
        matrice_nir[j, line] = band1[pixel[0], pixel[1]]
        matrice_ndvi[j, line] = ndvi[pixel[0], pixel[1]]
        matrice_ndwi[j, line] = ndwi[pixel[0], pixel[1]]
    line += 1

matrix_dic = {'vec_nir': matrice_nir, 'vec_red': matrice_rouge,
              'vec_green': matrice_vert, 'vec_blue': matrice_bleu,
              'vec_ndvi': matrice_ndvi, 'vec_ndwi': matrice_ndwi}

for matrix in matrix_dic:
    data = pd.DataFrame(matrix_dic[matrix])
    data.to_csv(OUT_DIR + f'/{matrix}.csv', index=False)
