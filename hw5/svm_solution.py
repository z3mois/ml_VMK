import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    model = SVC(kernel="rbf", C=2, gamma="auto", class_weight="balanced")
    scaler = StandardScaler()
    scaler.fit(train_features)
    model.fit(scaler.transform(train_features), train_target)
    pred = model.predict(scaler.transform(test_features))
    return pred
