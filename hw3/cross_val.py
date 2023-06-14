import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    i = 0
    folds = []
    div, mod = divmod(num_objects, num_folds)
    num_objects = [i for i in range(num_objects)]
    while i < len(num_objects):
        folds.append((num_objects[i:i+div], i))
        i = i + div
    answer = []
    for elem in folds:
        if mod != 0 and len(answer) == num_folds - 1:
            break
        temp = []
        for i in range(len(num_objects)):
            if abs(i - elem[1]) < div:
                if i < elem[1]:
                    temp.append((num_objects[i]))
            else:
                temp.append((num_objects[i]))
        answer.append((np.array(temp), np.array(elem[0])))
    if mod != 0:
        answer.append((np.array(num_objects[:(num_folds-1)*div]), np.array(num_objects[(num_folds-1)*div:])))
    return answer


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    answer = {}
    for n, m, w, normalize in [
                            (n, m, w, normalize) for n in parameters["n_neighbors"]
                            for m in parameters["metrics"]
                            for w in parameters["weights"]
                            for normalize in parameters["normalizers"]]:
        score = 0
        for item in folds:
            X_train = X[item[0]]
            X_val = X[item[1]]
            y_train = y[item[0]]
            y_val = y[item[1]]
            if normalize[0]:
                normalize[0].fit(X_train)
                X_train = normalize[0].transform(X_train)
                X_val = normalize[0].transform(X_val)
            model = knn_class(n_neighbors=n, metric=m, weights=w)
            model.fit(X=X_train, y=y_train)
            temp = model.predict(X_val)
            score += score_function(y_val, temp)
        score /= len(folds)
        answer[(normalize[1], n, m, w)] = score
    return answer

print(kfold_split(10, 3))