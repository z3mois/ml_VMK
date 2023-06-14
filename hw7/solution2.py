import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    distance_matrix = sklearn.metrics.pairwise_distances(x)
    num_clusters, count_clusters = np.unique(labels, return_counts=True)
    len_labels = len(labels)
    if len(num_clusters) <= 1:
        return 0
    masks = np.zeros((len_labels, len(num_clusters))).astype(bool)
    sum_dists = np.zeros((len_labels, len(num_clusters)))
    sizes_clusters = np.zeros(len_labels)
    for i, cluster in enumerate(num_clusters):
        masks[:, i] = labels == cluster
        sum_dists[:, i] = np.sum(distance_matrix[:, labels == cluster], axis=1)
        sizes_clusters[labels == cluster] = np.sum(labels == cluster)
    one_elem_cl = sizes_clusters == 1
    s = sum_dists[masks]
    s[one_elem_cl] = 0
    s[~one_elem_cl] /= (sizes_clusters[~one_elem_cl] - 1)
    d = np.min(((sum_dists / count_clusters)[~masks]).reshape(len_labels, -1), axis=1)
    d[one_elem_cl] = 0
    ans = np.zeros(len_labels)
    maxs = np.maximum(s, d)
    np.divide(d - s, maxs, out=ans, where=(maxs != 0))
    return np.mean(ans)


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    true_uniq, true_inv, true_count = np.unique(true_labels, return_inverse=True, return_counts=True)
    pred_uniq, pred_inv, pred_count = np.unique(predicted_labels, return_inverse=True, return_counts=True)
    true_labels[true_labels == 0] = true_uniq[-1] + 1
    predicted_labels[predicted_labels == 0] = pred_uniq[-1] + 1
    correctness = np.ones((len(true_labels), len(true_labels)))
    correctness[(true_labels / true_labels[:, None]) != 1] = 0
    correctness[(predicted_labels / predicted_labels[:, None]) != 1] = 0
    prec = np.mean(np.sum(correctness, axis=1) / pred_count[pred_inv])
    recall = np.mean(np.sum(correctness, axis=1) / true_count[true_inv])
    return 2 * prec * recall / (prec + recall)
