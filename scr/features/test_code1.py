import json

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

with open("data/pixels/lab_px_test_small.txt", "w") as fp:
    json.dump(dic_pix, fp)
print("File ready.")