import os
import re
import rasterio as rio
import pandas as pd
import numpy as np


def create_dic_pixels():
    '''
    Fonction qui permet de creer un dictionnaire avec les pixels de
    chaque type d'evolution.
    '''
    pix_foret = [[472, 570], [474, 570], [476, 570], [478, 570], [480, 570],
                [482, 570], [484, 570], [486, 570], [488, 570]]  # ca change uniformement (saison)
    pix_lac = [[392, 567], [392, 580], [401, 577], [401, 567], [395, 570], 
            [395, 576], [397, 571], [394, 598], [388, 532]]  # ca change un peu
    pix_apt = [[405, 448], [408, 444], [412, 446], [412, 463], [407, 465],
            [405, 455], [414, 440], [420, 458], [401, 446]]  # ca change (construction)
    pix_ensta = [[447, 618], [454, 627], [454, 631], [457, 632], [459, 625],
                [450, 641], [443, 636], [439, 629], [433, 617]]  # ca change (construction)
    pix_agri = [[318, 438], [322, 435], [324, 433], [329, 429], [333, 426],
                [337, 424], [339, 422], [344, 418], [350, 414]]  # ca peut changer (saison, plantation)
    pix_danone = [[383, 497], [383, 500], [387, 501], [383, 504], [387, 505],
                [384, 508], [388, 509], [384, 504], [386, 504]]  # ca ne change pas

    return {'pix_foret':pix_foret, 'pix_lac':pix_lac, 'pix_apt':pix_apt,  
            'pix_ensta':pix_ensta, 'pix_agri':pix_agri, 'pix_danone':pix_danone}


dic_pix = create_dic_pixels()
pixels_de_interet = [element for key in dic_pix.keys() for element in dic_pix[key]]


dir = "../../ressources/images"
images_list = os.listdir(dir)

# all images in a list, ready to be read
images_2A = list()
images_2B = list()
prog = re.compile(r'\w+2A')
for image in images_list:
    if prog.match(image):
        images_2A.append(image)
    else:
        images_2B.append(image)
        
images_2A.sort()
images_2B.sort()
all_images = images_2A + images_2B
all_images = sorted(all_images, key=lambda date: date[16:24])


# Initiate lists for data frame
date = []
pixel_r = []
pixel_g = []
pixel_b = []
pixel_ir = []
x_coord = []
y_coord = []

# read each image --> select interesting pixels --> add to list
for temp, img in zip(range(len(all_images)), all_images):
    img = dir + "/" + img
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

# the final DF has the values of pixels of interest for all the chanels and for all period of time
dic = {'date': date,
       'x_coord': x_coord,
       'y_coord': y_coord,
       'pixel_ir': pixel_ir,
       'pixel_r': pixel_r,
       'pixel_g': pixel_g,
       'pixel_b': pixel_b}

print(band_b)



