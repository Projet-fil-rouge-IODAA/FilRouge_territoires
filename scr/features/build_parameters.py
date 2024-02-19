import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
# from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from kneed import KneeLocator
import matplotlib.pyplot as plt

def find_num_clusters(data, model_type, k_min = 2, k_max = 10, plot_elbow = False, **kwargs):
    """
    This function finds the optimal number of clusters for a given dataset.
    """
    if model_type not in ['KMedoids', 'KMeans', 'Hierarchical', 'Spectral']:
        raise ValueError('{} is not supported'.format(model_type))
    else:
        sse = []
        if model_type == 'Hierarchical':
            model = shc.linkage(data, **kwargs)
            sse = model[-(k_max-k_min+1):, 2][::-1]
        elif model_type == 'Spectral':
            model = SpectralClustering(n_clusters=k_max, **kwargs)
            model.fit(data)
            sse, _ = np.linalg.eig(model.affinity_matrix_)
            sse = np.sort(sse)[::-1]
            sse = sse[k_min:k_max+1]
        else:
            for k in range(k_min, k_max+1):
                if model_type == 'KMedoids':
                    model = KMedoids(n_clusters=k, **kwargs)
                elif model_type == 'KMeans':
                    model = KMeans(n_clusters=k, **kwargs)
                model.fit(data)
                sse.append(model.inertia_)
        kl = KneeLocator(range(k_min, k_max+1), sse, curve='convex', direction='decreasing')
        if plot_elbow:
            kl.plot_knee(title=model_type, xlabel='Number of clusters', ylabel='SSE')
            plt.show()
        return round(kl.elbow)