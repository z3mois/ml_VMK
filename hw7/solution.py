import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    # Ваш код здесь:＼(º □ º l|l)/

    clust = np.unique(labels, return_counts=True)[1]
    if len(clust) == 1: return 0

    x = x[np.argsort(labels)]

    dist = sklearn.metrics.pairwise.pairwise_distances(x, metric='euclidean', n_jobs=-1)

    first = 0
    correct = []
    dists = np.zeros((len(labels), len(clust)))

    for i in range(len(clust)):
        cnt = clust[i]
        last = first + cnt
        dist_i = dist[:, first:last]
        dists[:, i] = np.sum(dist_i, axis=1)
        first += cnt
        correct = correct + [i] * cnt

    first = np.arange(len(dists))
    last = correct

    s = dists[first, last]
    s = np.divide(s, clust[correct] - 1, where=clust[correct] != 1, out=np.zeros_like(labels, dtype=float))

    dists[first, last] = np.inf

    min_i = np.argmin(dists, axis=1)
    first = np.arange(dists.shape[0])
    d = dists[first, min_i] / clust[min_i]

    max_d = np.maximum(d, s)
    res = np.divide(d - s, max_d, where=max_d != 0, out=np.zeros_like(labels, dtype=float))
    res[clust[correct] == 1] = 0
    res = res.mean()
    return res


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    # Ваш код здесь:＼(º □ º l|l)/

    true = true_labels[:, np.newaxis] == true_labels
    pred = predicted_labels[:, np.newaxis] == predicted_labels
    true_pred = true * pred

    true = np.sum(true, axis=1)
    pred = np.sum(pred, axis=1)
    true_pred = np.sum(true_pred, axis=1)

    precision = np.mean(true_pred / pred)
    recall = np.mean(true_pred / true)

    B_Cubed = 2 * (precision * recall) / (precision + recall)

    return B_Cubed
