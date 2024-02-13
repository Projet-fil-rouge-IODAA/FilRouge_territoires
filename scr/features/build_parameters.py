import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from kneed import KneeLocator
import matplotlib.pyplot as plt

def find_num_clusters(data, k_min, k_max, model_type, **kwargs):
    """
    This function finds the optimal number of clusters for a given dataset.
    """
    if model_type not in ['KMedoids', 'KMeans', 'Hierarchical']:
        raise ValueError('{} is not supported'.format(model_type))
    else:
        sse = []
        klist = np.arange(k_min, k_max+1, 1)
        if model_type == 'Hierarchical':
            model = shc.linkage(data, **kwargs)
                # model = AgglomerativeClustering(n_clusters=k)
                # model.fit(data)
            sse = model[-10:, 2][::-1]
            klist = np.arange(k_min, k_max+1, 2)
        else:
            for k in range(k_min, k_max+1):
                if model_type == 'KMedoids':
                    model = KMedoids(n_clusters=k, **kwargs)
                elif model_type == 'KMeans':
                    model = KMeans(n_clusters=k, **kwargs)
                model.fit(data)
                sse.append(model.inertia_)
        kl = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
        kl.plot_knee(title=model_type, xlabel='Number of clusters', ylabel='SSE')
        plt.show()
        return round(kl.elbow)