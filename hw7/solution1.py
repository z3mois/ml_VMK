import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    number_of_clusters, _ = np.unique(labels, return_counts=True)
    if len(number_of_clusters) == 1:
        return 0
    sort_idx = np.argsort(labels)
    sorted_x = x[sort_idx]

    pairwise_distances_matrix = sklearn.metrics.pairwise.pairwise_distances(sorted_x, metric='euclidean', n_jobs=-1)

    idx0 = 0
    correct_labels = []
    clusters_pairwise_distances = np.zeros((len(labels), len(number_of_clusters)))
    for i in range(len(number_of_clusters)):
        cnt = number_of_clusters[i]
        idx1 = idx0 + cnt
        pairwise_distances_for_i_cluster = pairwise_distances_matrix[:, idx0:idx1]
        clusters_pairwise_distances[:, i] = np.sum(pairwise_distances_for_i_cluster, axis=1)
        idx0 += cnt
        correct_labels = correct_labels + [i] * cnt
    idx_0_for_s = np.arange(len(clusters_pairwise_distances))
    idx_1_for_s = correct_labels
    new_s = clusters_pairwise_distances[idx_0_for_s, idx_1_for_s]
    s = np.divide(new_s, number_of_clusters[correct_labels] - 1, where=number_of_clusters[correct_labels] != 1, out=np.zeros_like(labels, dtype=float))
    clusters_pairwise_distances[idx_0_for_s, idx_1_for_s] = np.inf
    idx_for_min_dists = np.argmin(clusters_pairwise_distances, axis=1)
    idx_0_for_d = np.arange(clusters_pairwise_distances.shape[0])
    new_d = clusters_pairwise_distances[idx_0_for_d, idx_for_min_dists]
    d = new_d / number_of_clusters[idx_for_min_dists]
    maximum_d_s = np.maximum(d, s)
    res = np.divide(d - s, maximum_d_s, where=maximum_d_s != 0, out=np.zeros_like(labels, dtype=float))
    res[number_of_clusters[correct_labels] == 1] = 0
    res = res.mean()
    return res


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    Model_answers = predicted_labels[:, np.newaxis] == predicted_labels
    Gold_standart = true_labels[:, np.newaxis] == true_labels
    Correctness = Model_answers * Gold_standart
    Correctness_summ = np.sum(Correctness, axis=1)
    Model_answers_summ = np.sum(Model_answers, axis=1)
    Gold_standart_summ = np.sum(Gold_standart, axis=1)
    Precision = np.mean(Correctness_summ / Model_answers_summ)
    Recall = np.mean(Correctness_summ / Gold_standart_summ)
    B_Cubed = 2 * (Precision * Recall) / (Precision + Recall)
    return B_Cubed
