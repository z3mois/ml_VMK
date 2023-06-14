import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.max = np.amax(data, axis=0)
        self.min = np.amin(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        answer = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                answer[i][j] = (data[i][j] - self.min[j]) / (self.max[j] - self.min[j])
        return answer


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.var = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        answer = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                answer[i][j] = (data[i][j] - self.mean[j]) / self.var[j]
        return answer
