import os
import re
import rasterio as rio
import pandas as pd
import numpy as np
import pickle
import shutil
import json
import sys 
import torch

IN_DIR = "data/cropped"
OUT_DIR = "data/processed"

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

# making the output directory
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.mkdir(OUT_DIR)
name = input("What type of pixel sample do you wish to use? (big, small, or all) : ")

with_texture = input("Do you want to add texture bands to the images? (y or n) : ")
with_texture = (with_texture == 'y')

if name == 'all':
    x,y = np.load(IN_DIR + "/" + all_images[0])[0].shape
    pixels_de_interet = []
    for i in range(x):
        for j in range(y):
            pixels_de_interet.append([i,j])
elif name == 'big' or name == 'small':
    name_file = "data/pixels/lab_px_test_"+name+".txt"
else:
    raise Exception("Sorry, sample name other than big, small or all.")

def create_dic_pixels(name_file):
    '''
    Fonction qui permet de creer un dictionnaire avec les pixels de
    chaque type d'evolution.
    '''
    data =  np.load(name_file)

    dic = json.loads(data)
    list = []
    for v in dic.values():
        list.extend(v)

    return list, dic

if name == 'big' or name == 'small':
    pixels_de_interet, dic_pix = create_dic_pixels(name_file)

    # Saving the list and the dictionary as pickle objects.
    with open(OUT_DIR + '/pixels_de_interet_dic.pkl', 'wb') as file:
        pickle.dump(dic_pix, file)

with open(OUT_DIR + '/pixels_de_interet_list.pkl', 'wb') as file:
    pickle.dump(pixels_de_interet, file)



# Initiate lists to compose the data frame.
date = []
pixel_r = []
pixel_g = []
pixel_b = []
pixel_ir = []
pixel_e = []
pixel_h = []
x_coord = []
y_coord = []

# read each image --> select interesting pixels --> add to list
for temp, img in zip(range(len(all_images)), all_images):
    img = IN_DIR + "/" + img
    img = np.load(img)
    band_ir = img[0]
    band_r = img[1]
    band_g = img[2]
    band_b = img[3]
    if with_texture:
        band_e = img[4]
        band_h = img[5]
    

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

        if with_texture:
            e = band_e[px_x, px_y]
            h = band_h[px_x, px_y]
            pixel_e.append(e)
            pixel_h.append(h)

print("Creating the dict...")

# The final DF has the values for all the bands and time periods.
dic = {'date': date,
       'x_coord': x_coord,
       'y_coord': y_coord,
       'pixel_ir': pixel_ir,
       'pixel_r': pixel_r,
       'pixel_g': pixel_g,
       'pixel_b': pixel_b}

if with_texture:
    dic['pixel_e'] = pixel_e
    dic['pixel_h'] = pixel_h

data = pd.DataFrame(dic)
data = torch.tensor(data)
torch.save(data, OUT_DIR + "/by_date_and_coord.pt")

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
if with_texture:
    matrice_energy = np.zeros((len(pixels_de_interet), len(all_images)))
    matrice_homogeneity = np.zeros((len(pixels_de_interet), len(all_images)))

for image in all_images:
    ds = np.load(IN_DIR + '/' + image)
    band1 = ds[0]  # Near Infrared.
    band2 = ds[1]  # Red.
    band3 = ds[2]  # Green.
    band4 = ds[3]  # Blue.
    if with_texture:
        band5 = ds[4]  # Texture Energy.
        band6 = ds[5]  # Texture Homogeneity.
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
        if with_texture:
            matrice_energy[j, line] = band5[pixel[0], pixel[1]]
            matrice_homogeneity[j, line] = band6[pixel[0], pixel[1]]
    line += 1

matrix_dic = {'vec_nir': matrice_nir, 'vec_red': matrice_rouge,
              'vec_green': matrice_vert, 'vec_blue': matrice_bleu,
              'vec_ndvi': matrice_ndvi, 'vec_ndwi': matrice_ndwi}
if with_texture:
    matrix_dic['vec_energy'] = matrice_energy
    matrix_dic['vec_homogeneity'] = matrice_homogeneity

for matrix in matrix_dic:
    data = pd.DataFrame(matrix_dic[matrix])
    data.to_csv(OUT_DIR + f'/{matrix}.csv', index=False)

# --------------------- DFs for T2F ----------------------

array3d = np.zeros((matrice_vert.shape[0], matrice_vert.shape[1], len(matrix_dic)))
for i, matrix in enumerate(list(matrix_dic.values())):
    array3d[:, :, i] = matrix
arr_reshaped = array3d.reshape(array3d.shape[0], -1)
np.savetxt(OUT_DIR + f'/{array3d.shape}.csv', arr_reshaped, delimiter=',')
