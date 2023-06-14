import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import numpy as np


def recenter(arr):
    import scipy as sp
    slicing = sp.ndimage.find_objects(arr != 0, max_label=1)[0]
    center_slicing = tuple(
        slice((dim - sl.stop + sl.start) // 2, (dim + sl.stop - sl.start) // 2)
        for sl, dim in zip(slicing, arr.shape))
    result = np.zeros_like(arr)
    result[center_slicing] = arr[slicing]
    return result


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """
    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        for idx, _ in enumerate(x):
            x[idx] -= 20
            x[idx] = recenter(x[idx])
        x_train = np.array([np.array(list(map(sum, zip(*elem)))) for elem in x])
        # return x.reshape((x.shape[0], -1))
        return x_train


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    p = PotentialTransformer()
    X_train = p.fit_transform(X_train, Y_train)
    X_test = p.fit_transform(X_test, Y_train)
    regressor = ExtraTreesRegressor(max_features="sqrt", max_depth = 10, criterion="friedman_mse", n_estimators = 3000)
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
