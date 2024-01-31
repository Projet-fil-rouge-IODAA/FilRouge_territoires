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
    pix_test = []

    dic = {'pix_test': pix_test}
    list = pix_test

    return list, dic


pixels_de_interet, dic_pix = create_dic_pixels()

# TODO
