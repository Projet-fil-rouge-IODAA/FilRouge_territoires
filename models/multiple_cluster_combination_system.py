# Nada et Julian 06/02/2024


def voating_system(*args): 
    '''
    def de matrice : 
    [
    [[x1,x2], .......... ]
    [                    ]    
    [                    ]

    ]
    '''
    secondary_c = args[1:]
    n_clusters = 3
    matrice  = [[[] for _ in range(len(set(yhat_rouge)))] for _ in range(len(set(yhat_vert))+len(set(yhat_bleu)))]
    for i in range(len(yhat_rouge)):
        matrice[yhat_vert[i]][yhat_rouge[i]].append(i)
        matrice[len(set(yhat_vert))+yhat_bleu[i]][yhat_rouge[i]].append(i)
    # display(matrice)
    # ----------------- Matrice de confusion --------------------------------
    conf = [[len(matrice[j][i]) for i in range(len(set(yhat_rouge)))] for j in range(len(set(yhat_vert))+len(set(yhat_bleu)))]
    conf = np.array(conf).astype(dtype=np.float16)
    display(conf)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            conf[i,j]=conf[i, j]/(max(conf[i,:].sum(), (conf[:,j].sum()/(n_clusters-1))))

    display(conf)