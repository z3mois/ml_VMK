import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        answer = []
        x = X.to_numpy()
        for j in range(x.shape[1]):
            temp = []
            for i in range(x.shape[0]):
                if x[i][j] not in temp:
                    temp.append(x[i][j])
            answer.append(sorted(temp))
        self.unique = answer

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        answer = None
        x = X.to_numpy()
        for j in range(x.shape[1]):
            temp = np.zeros((x.shape[0], len(self.unique[j])))
            for i in range(x.shape[0]):
                for index, elem in enumerate(self.unique[j]):
                    if elem == x[i][j]:
                        temp[i][index] = 1
                        break
            if answer is None:
                answer = temp
            else:
                answer = np.concatenate((answer, temp), axis=1)
        return answer

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        x = X.to_numpy()
        y = Y.to_numpy()
        answer = []
        for j in range(x.shape[1]):
            temp = {}
            for i in range(x.shape[0]):
                if x[i][j] not in temp:
                    counters = 0
                    successes = 0
                    for i1 in range(x.shape[0]):
                        if x[i1][j] == x[i][j]:
                            counters += 1
                            successes += y[i1]
                    successes = successes / counters
                    counters = counters / x.shape[0]
                    temp[x[i][j]] = np.array([successes, counters, 0])
            answer.append(temp)
        self.dictinory = answer

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        answer = None
        x = X.to_numpy()
        for j in range(x.shape[1]):
            temp = np.zeros((x.shape[0], 3))
            for i in range(x.shape[0]):
                temp[i] = self.dictinory[j][x[i][j]]
                temp[i][2] = (temp[i][0] + a) / (temp[i][1] + b)
            if answer is None:
                answer = temp
            else:
                answer = np.concatenate((answer, temp), axis=1)
        return answer

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        # your code here
        self.groups = group_k_fold(X.shape[0], self.n_folds, seed)
        answer = []
        for val, train in self.groups:
            sce = SimpleCounterEncoder()
            sce.fit(X.iloc[train], Y.iloc[train])
            answer.append((val, sce))
        self.answer = answer

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        # your code here
        temp = None
        for val, class1 in self.answer:
            print(type(val))
            con = np.concatenate((class1.transform(X.iloc[val], a, b), np.reshape(np.array(val), (len(val), 1))), axis=1)
            if temp is None:
                temp = con
            else:
                temp = np.concatenate((temp, con), axis=0)
        temp = temp[temp[:, -1].argsort()]
        return np.delete(temp, -1, 1)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    # your code her
    set_temp = set(x)
    w = np.array([0.0]*len(set_temp))
    for i, elem in enumerate(set_temp):
        w[i] = sum(y[x == elem]) / list(x).count(elem)
    return w
