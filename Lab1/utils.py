import numpy as np
import neuralNetwork 


class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class dataSet:
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def iterate(self, batch_size):
        num_batches = self.x.shape[0] // batch_size  # 计算可完整迭代的批次数
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x = self.x[start_idx:end_idx]
            y = self.y[start_idx:end_idx]
            yield neuralNetwork.Constant(x), neuralNetwork.Constant(y)

