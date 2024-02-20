import numpy as np
import scipy.spatial.distance as dist
import fastdtw as dtw


def distance(ts1: np.array, ts2: np.array, distance_feat: str = None):
    metrics = {
        'braycurtis': dist.braycurtis,
        'canberra': dist.canberra,
        'chebyshev': dist.chebyshev,
        'cityblock': dist.cityblock,
        'correlation': dist.correlation,
        'cosine': dist.cosine,
        'euclidean': dist.euclidean,
        'minkowski': dist.minkowski,
        'dtw': dtw.fastdtw,
        # 'mahalanobis': dist.mahalanobis,
        # 'seuclidean': dist.seuclidean,
        # 'sqeuclidean': dist.sqeuclidean,
    }
    if distance_feat is None:
        distances = {k: f(ts1, ts2, dist=dist.euclidean)[0] if k == 'dtw' else f(ts1, ts2) for k, f in metrics.items()}
    else:
        distances = {distance_feat: metrics[distance_feat](ts1, ts2) if distance_feat != 'dtw' else dtw.fastdtw(ts1, ts2, dist = dist.euclidean)[0]}
    return distances


def extract_pair_features(ts1: np.array, ts2: np.array, distance_feat: str = None):
    assert len(ts1.shape) == 1, 'ts1 is not a univariate time series'
    assert len(ts2.shape) == 1, 'ts2 is not a univariate time series'

    features = {}
    distances = distance(ts1, ts2, distance_feat)
    features.update(distances)

    return features
