import numpy as np
import neuralNetwork as nn
class dataSet:
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(x.dtype, np.float32)
        assert np.issubdtype(y.dtype, np.float32)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def iterate(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, expected int, got {!r}".format(
                batch_size))
        assert self.x.shape[0] % batch_size == 0, (
            "Dataset size {:d} is not divisible by batch size {:d}".format(
                self.x.shape[0], batch_size))
        for i in range(0, batch_size, self.x.shape[0]):
            x = self.x[i: i + batch_size]
            y = self.y[i: i + batch_size]
            yield nn.Constant(x), nn.Constant(y)