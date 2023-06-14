from catboost import CatBoostRegressor
import pandas as pd
from numpy import ndarray


class Transformer:
    dict_key_words = {}
    genres = None
    direct = None
    film_loc = None
    bound = 100

    def fit(self, X, num_bound):
        y = X['awards']
        X = X.drop(columns='awards')
        lst = [b for i in list(X['keywords']) for b in i]
        dict_ = dict(zip(lst, [lst.count(i) for i in lst]))
        self.dict_key_words = dict(sorted(dict_.items(), key=lambda item: item[1])[::-1])
        self.bound = num_bound
        self.genres = list(X['genres'])
        self.direct = list(X['directors'])
        self.film_loc = list(X['filming_locations'])
        return X, y

    def transform(self, X):
        X[[b for i in self.genres for b in i]] = 0
        X[[b for i in self.direct for b in i]] = 0
        X[[b for i in self.film_loc for b in i]] = 0
        for key, value in self.dict_key_words.items():
            if value < self.bound:
                break
            X[key] = 0
        for val in {b for i in list(X['keywords']) for b in i}:
            if val in self.dict_key_words.keys() and self.dict_key_words[val] >= self.bound:
                X.loc[[val in row['keywords'] for i, row in X.iterrows()], val] = 1

        list_ = ['genres', 'directors', 'filming_locations']
        for k in list_:
            X[[b for i in list(X[k]) for b in i]] = 0
            for val in {b for i in list(X[k]) for b in i}:
                X.loc[[val in row[k] for i, row in X.iterrows()], val] = 1

            X = X.drop(columns=k)

        X['actor_0_gender'] = X['actor_0_gender'].astype('category')
        X['actor_1_gender'] = X['actor_1_gender'].astype('category')
        X['actor_2_gender'] = X['actor_2_gender'].astype('category')

        X = X.drop(columns=['u', 'n', 'k', 'w', 'o'])
        X = X.drop(columns='keywords')
        return X

    def fit_transform(self, X, num_bound):
        X, y = self.fit(X, num_bound)
        return self.transform(X), y


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    X_train = pd.read_json(train_file, lines=True)
    X_test = pd.read_json(test_file, lines=True)
    tr = Transformer()
    X_train, y_train = tr.fit_transform(X_train, 58)
    X_test = tr.transform(X_test)
    catg = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']
    regressor = CatBoostRegressor(**{'learning_rate': 0.04626808450407065,
                                     'max_depth': 9, 'n_estimators': 455}, cat_features=catg, verbose=False, train_dir="/tmp/catboost_info")
    regressor.fit(X_train, y_train)
    return regressor.predict(X_test)
